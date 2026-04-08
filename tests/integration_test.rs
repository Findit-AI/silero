use silero::{BatchInput, SampleRate, Session, SpeechOptions, StreamState, detect_speech};

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

#[test]
fn bundled_session_loads() {
  let _session = Session::bundled().expect("bundled model should load");
}

#[test]
fn silence_settles_to_low_probability() {
  let mut session = Session::bundled().expect("bundled model should load");
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
  let mut single_session = Session::bundled().expect("bundled model should load");
  let mut batch_session = Session::bundled().expect("bundled model should load");
  let mut single_a = StreamState::new(SampleRate::Rate16k);
  let mut single_b = StreamState::new(SampleRate::Rate16k);
  let mut batch_a = StreamState::new(SampleRate::Rate16k);
  let mut batch_b = StreamState::new(SampleRate::Rate16k);

  let audio_a = pseudo_audio(SampleRate::Rate16k.chunk_samples() * 6);
  let audio_b = pseudo_audio(SampleRate::Rate16k.chunk_samples() * 6 + 13);

  let expected_a: Vec<f32> = audio_a
    .chunks_exact(SampleRate::Rate16k.chunk_samples())
    .map(|chunk| {
      single_session
        .infer_chunk(&mut single_a, chunk)
        .expect("single infer a")
    })
    .collect();
  let expected_b: Vec<f32> = audio_b[..SampleRate::Rate16k.chunk_samples() * 6]
    .chunks_exact(SampleRate::Rate16k.chunk_samples())
    .map(|chunk| {
      single_session
        .infer_chunk(&mut single_b, chunk)
        .expect("single infer b")
    })
    .collect();

  let mut actual_a = Vec::new();
  let mut actual_b = Vec::new();
  for (chunk_a, chunk_b) in audio_a
    .chunks_exact(SampleRate::Rate16k.chunk_samples())
    .zip(
      audio_b[..SampleRate::Rate16k.chunk_samples() * 6]
        .chunks_exact(SampleRate::Rate16k.chunk_samples()),
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
  let mut session = Session::bundled().expect("bundled model should load");
  let mut stream = StreamState::new(SampleRate::Rate16k);
  let audio = pseudo_audio(SampleRate::Rate16k.chunk_samples() * 3 + 200);
  let mut probabilities = Vec::new();

  let processed = session
    .process_stream(&mut stream, &audio, |probability| {
      probabilities.push(probability)
    })
    .expect("process stream");
  assert_eq!(processed, 3);
  assert!(stream.has_pending());

  if let Some(probability) = session.flush_stream(&mut stream).expect("flush stream") {
    probabilities.push(probability);
  }

  assert_eq!(probabilities.len(), 4);
  assert!(!stream.has_pending());
}

#[test]
fn detect_speech_on_silence_returns_empty() {
  let mut session = Session::bundled().expect("bundled model should load");
  let audio = vec![0.0_f32; SampleRate::Rate16k.chunk_samples() * 8];
  let segments = detect_speech(
    &mut session,
    &audio,
    SpeechOptions::default().with_sample_rate(SampleRate::Rate16k),
  )
  .expect("detect speech");
  assert!(segments.is_empty());
}
