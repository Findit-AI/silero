// The integration suite exercises the ONNX-backed `Session` API, which
// only exists with the default `onnx` feature. Under
// `--no-default-features` the crate is logic-only, so this file compiles
// to an empty test target.
#![cfg(feature = "onnx")]

use silero::{
  BatchInput, SampleRate, Session, SpeechOptions, SpeechSegmenter, SpeechSegmenterExt, StreamState,
  detect_speech, detect_speech_with,
};

const MODEL_BYTES: &[u8] = include_bytes!(concat!(
  env!("CARGO_MANIFEST_DIR"),
  "/models/silero_vad.onnx"
));

fn test_session() -> Session {
  Session::from_memory(MODEL_BYTES).expect("test model should load")
}

fn pseudo_audio(len: usize) -> Vec<f32> {
  let mut value = 0x1234_5678_u32;
  let mut out = Vec::with_capacity(len);
  for _ in 0..len {
    value = value.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    let sample = ((value >> 8) as f32 / ((u32::MAX >> 8) as f32)) * 2.0 - 1.0;
    out.push(sample * 0.15);
  }
  out
}

#[cfg(feature = "bundled")]
#[test]
fn bundled_session_loads() {
  let _session = Session::bundled().expect("bundled model should load");
}

#[test]
fn silence_settles_to_low_probability() {
  let mut session = test_session();
  let mut stream = StreamState::new(SampleRate::Rate16k);
  let silence = vec![0.0_f32; SampleRate::Rate16k.chunk_samples()];

  let mut last = 1.0;
  for _ in 0..40 {
    last = session
      .infer_chunk(&mut stream, &silence)
      .expect("infer chunk");
  }

  assert!(last < 0.05, "expected near-silence probability, got {last}");
}

#[test]
fn batch_inference_matches_single_stream_inference() {
  let mut single_session = test_session();
  let mut batch_session = test_session();
  let mut single_a = StreamState::new(SampleRate::Rate16k);
  let mut single_b = StreamState::new(SampleRate::Rate16k);
  let mut batch_a = StreamState::new(SampleRate::Rate16k);
  let mut batch_b = StreamState::new(SampleRate::Rate16k);

  let audio_a = pseudo_audio(SampleRate::Rate16k.chunk_samples() * 6);
  let audio_b = pseudo_audio(SampleRate::Rate16k.chunk_samples() * 6 + 13);

  let expected_a: Vec<f32> = audio_a
    .as_chunks::<{ SampleRate::Rate16k.chunk_samples() }>()
    .0
    .iter()
    .map(|chunk| {
      single_session
        .infer_chunk(&mut single_a, chunk)
        .expect("single infer a")
    })
    .collect();
  let expected_b: Vec<f32> = audio_b[..SampleRate::Rate16k.chunk_samples() * 6]
    .as_chunks::<{ SampleRate::Rate16k.chunk_samples() }>()
    .0
    .iter()
    .map(|chunk| {
      single_session
        .infer_chunk(&mut single_b, chunk)
        .expect("single infer b")
    })
    .collect();

  let mut actual_a = Vec::new();
  let mut actual_b = Vec::new();
  for (chunk_a, chunk_b) in audio_a
    .as_chunks::<{ SampleRate::Rate16k.chunk_samples() }>()
    .0
    .iter()
    .zip(
      audio_b[..SampleRate::Rate16k.chunk_samples() * 6]
        .as_chunks::<{ SampleRate::Rate16k.chunk_samples() }>()
        .0
        .iter(),
    )
  {
    let mut batch = [
      BatchInput::new(&mut batch_a, chunk_a),
      BatchInput::new(&mut batch_b, chunk_b),
    ];
    let probabilities = batch_session
      .infer_batch(&mut batch)
      .expect("batched infer");
    actual_a.push(probabilities[0]);
    actual_b.push(probabilities[1]);
  }

  assert_eq!(expected_a.len(), actual_a.len());
  assert_eq!(expected_b.len(), actual_b.len());
  for (expected, actual) in expected_a.iter().zip(actual_a.iter()) {
    assert!((expected - actual).abs() < 1e-6);
  }
  for (expected, actual) in expected_b.iter().zip(actual_b.iter()) {
    assert!((expected - actual).abs() < 1e-6);
  }
}

#[test]
fn process_stream_and_flush_cover_partial_tail() {
  let mut session = test_session();
  let mut stream = StreamState::new(SampleRate::Rate16k);
  let audio = pseudo_audio(SampleRate::Rate16k.chunk_samples() * 3 + 200);
  let mut probabilities = Vec::new();

  probabilities.extend_from_slice(
    session
      .process_stream(&mut stream, &audio)
      .expect("process stream"),
  );
  assert_eq!(probabilities.len(), 3);
  assert!(stream.has_pending());

  if let Some(probability) = session.flush_stream(&mut stream).expect("flush stream") {
    probabilities.push(probability);
  }

  assert_eq!(probabilities.len(), 4);
  assert!(!stream.has_pending());
}

#[test]
fn detect_speech_on_silence_returns_empty() {
  let mut session = test_session();
  let audio = vec![0.0_f32; SampleRate::Rate16k.chunk_samples() * 8];
  let segments = detect_speech(
    &mut session,
    &audio,
    SpeechOptions::default().with_sample_rate(SampleRate::Rate16k),
  )
  .expect("detect speech");
  assert!(segments.is_empty());
}

#[test]
fn finish_stream_clears_active_state_so_a_followup_finish_does_not_re_emit() {
  // Pin in 0.4.0: `finish_stream` must clear the in-flight segment
  // tracker after enqueueing the trailing segment so `is_active()`
  // reflects end-of-stream and a follow-up `finish_stream` (or
  // `finish`) can't re-emit the same segment. Caught in PR #6
  // review (Copilot).
  use std::time::Duration;

  let mut session = test_session();
  let mut stream = StreamState::new(SampleRate::Rate16k);
  // Big enough that the trailing audio confirms an open segment.
  let chunk = SampleRate::Rate16k.chunk_samples();
  let mut audio = vec![0.0_f32; chunk * 2];
  audio.extend(pseudo_audio(chunk * 32));
  let config = SpeechOptions::default()
    .with_sample_rate(SampleRate::Rate16k)
    .with_min_speech_duration(Duration::ZERO);
  let mut segmenter = SpeechSegmenter::new(config);

  // Feed enough samples to keep an active segment open.
  let _ = segmenter
    .push_samples(&mut session, &mut stream, &audio)
    .expect("push samples");
  // Drain anything queued.
  while segmenter
    .push_samples(&mut session, &mut stream, &[])
    .expect("drain")
    .is_some()
  {}

  // First finish_stream: emits trailing if active.
  let mut emitted_first = 0;
  if let Some(_seg) = segmenter
    .finish_stream(&mut session, &mut stream)
    .expect("finish_stream first call")
  {
    emitted_first += 1;
    while segmenter
      .push_samples(&mut session, &mut stream, &[])
      .expect("drain after finish 1")
      .is_some()
    {
      emitted_first += 1;
    }
  }

  // After finish_stream, no segment is active and the queue is drained.
  assert!(
    !segmenter.is_active(),
    "is_active() must be false after finish_stream"
  );
  assert_eq!(segmenter.pending_segment_count(), 0);

  // A follow-up finish_stream / finish must NOT re-emit a segment.
  let second = segmenter
    .finish_stream(&mut session, &mut stream)
    .expect("finish_stream second call");
  assert!(
    second.is_none(),
    "second finish_stream must not re-emit (was {:?}, first emitted {})",
    second,
    emitted_first
  );
  let trailing = segmenter.finish();
  assert!(trailing.is_none(), "follow-up finish() must not re-emit");
}

#[test]
fn detect_speech_with_over_session_backend_matches_detect_speech() {
  // The generic `zuoer::detect_speech_with` driving the `Session` as a
  // `VadBackend` (push + finish) must agree with silero's own
  // `detect_speech` (push_samples + finish_stream) on the same
  // non-chunk-aligned buffer: both feed the identical inference and apply
  // Silero's zero-pad end-of-stream policy to the trailing partial frame.
  // This is the cross-check that the redesigned backend seam produces the
  // same segments as the bundled helper — and it exercises
  // `VadBackend::push` AND `VadBackend::finish` (the padded-tail path) over
  // the real model.
  let mut backend_session = test_session();
  let mut helper_session = test_session();
  // 3 full chunks plus a half-chunk tail that finish() must zero-pad.
  let chunk = SampleRate::Rate16k.chunk_samples();
  let audio = pseudo_audio(chunk * 3 + chunk / 2);
  let config = SpeechOptions::default().with_sample_rate(SampleRate::Rate16k);

  let via_generic =
    detect_speech_with(&mut backend_session, &audio, config.clone()).expect("detect_speech_with");
  let via_helper = detect_speech(&mut helper_session, &audio, config).expect("detect_speech");

  assert_eq!(
    via_generic, via_helper,
    "the generic VadBackend seam (push + finish) must match the bundled detect_speech helper"
  );
}
