//! Integration tests driving the Silero VAD ONNX model on real audio.
//!
//! These tests intentionally skip when either the model file or the
//! audio fixtures are missing, so the crate still builds and its
//! unit tests still run on a bare clone without any large binary
//! assets. To execute the full suite, run from the crate root (the
//! model file is bundled in the repo under `models/`):
//!
//! ```bash
//! RUSTFLAGS="-L /path/to/libonnxruntime-dir" \
//!     cargo test --release --test integration_test -- --nocapture
//! ```
//!
//! Audio fixture paths can be overridden with environment variables.

use std::path::{Path, PathBuf};

use silero::{
  speech_timestamps, speech_timestamps_with_model, SampleRate, SileroModel, VadConfig, VadIterator,
};

// ── Fixtures ────────────────────────────────────────

fn model_path() -> PathBuf {
  PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models/silero_vad.onnx")
}

fn tram_path() -> PathBuf {
  std::env::var("SILERO_TRAM_AUDIO")
    .map(PathBuf::from)
    .unwrap_or_else(|_| PathBuf::from("/tmp/tram_16k.wav"))
}

fn dialog_path() -> PathBuf {
  std::env::var("SILERO_DIALOG_AUDIO")
    .map(PathBuf::from)
    .unwrap_or_else(|_| PathBuf::from("/tmp/dialog_2person.wav"))
}

fn read_wav_16k_mono(path: &Path) -> Option<Vec<f32>> {
  if !path.exists() {
    eprintln!("Skipping: fixture not found at {}", path.display());
    return None;
  }
  let reader = hound::WavReader::open(path).ok()?;
  let spec = reader.spec();
  if spec.channels != 1 || spec.sample_rate != 16_000 {
    eprintln!(
      "Skipping: fixture {} is not 16 kHz mono ({} ch, {} Hz)",
      path.display(),
      spec.channels,
      spec.sample_rate
    );
    return None;
  }
  let samples: Vec<f32> = match spec.sample_format {
    hound::SampleFormat::Int => {
      let max = (1_i64 << (spec.bits_per_sample - 1)) as f32;
      reader
        .into_samples::<i32>()
        .filter_map(|s| s.ok().map(|v| v as f32 / max))
        .collect()
    }
    hound::SampleFormat::Float => reader
      .into_samples::<f32>()
      .filter_map(|s| s.ok())
      .collect(),
  };
  Some(samples)
}

fn skip_if_missing_model() -> bool {
  if !model_path().exists() {
    eprintln!(
      "Skipping: Silero ONNX model not found at {}",
      model_path().display()
    );
    return true;
  }
  false
}

// ── Tests ───────────────────────────────────────────

/// `SileroModel::from_path` loads the bundled ONNX without panicking.
#[test]
fn model_loads_from_bundled_path() {
  if skip_if_missing_model() {
    return;
  }
  let model = SileroModel::from_path(model_path()).expect("model loads");
  assert_eq!(model.sample_rate(), SampleRate::Rate16k);
  assert_eq!(model.chunk_samples(), 512);
}

/// Processing 512 samples of silence at 16 kHz yields a near-zero
/// voice probability. This is the most basic "is the inference
/// actually running" smoke test.
#[test]
fn process_silence_returns_low_probability() {
  if skip_if_missing_model() {
    return;
  }
  let mut model = SileroModel::from_path(model_path()).expect("model loads");
  let silence = vec![0.0_f32; 512];
  // First few chunks may drift because the LSTM has to settle.
  // Process 40 chunks (~1.3 s) and check the final one.
  let mut last = 1.0;
  for _ in 0..40 {
    last = model.process(&silence).expect("process");
  }
  assert!(
    last < 0.05,
    "silence should have near-zero voice probability after warmup, got {last}"
  );
}

/// Wrong chunk size returns a typed error rather than panicking.
#[test]
fn process_rejects_wrong_chunk_size() {
  if skip_if_missing_model() {
    return;
  }
  let mut model = SileroModel::from_path(model_path()).expect("model loads");
  let err = model.process(&vec![0.0_f32; 300]).unwrap_err();
  assert!(matches!(err, silero::SileroError::InvalidInput(_)));
}

/// Resetting the model between streams clears both LSTM state and
/// rolling context. We verify this by processing real dialog audio,
/// then processing the same audio again after reset, and checking
/// that both runs produce the same probabilities.
#[test]
fn reset_makes_runs_deterministic() {
  if skip_if_missing_model() {
    return;
  }
  let Some(dialog) = read_wav_16k_mono(&dialog_path()) else {
    return;
  };

  let mut model = SileroModel::from_path(model_path()).expect("model loads");

  // First run on 5 seconds of audio.
  let chunk = 512;
  let slice = &dialog[..5 * 16_000];
  let run1: Vec<f32> = slice
    .chunks_exact(chunk)
    .map(|c| model.process(c).unwrap())
    .collect();

  // Reset and re-run.
  model.reset();
  let run2: Vec<f32> = slice
    .chunks_exact(chunk)
    .map(|c| model.process(c).unwrap())
    .collect();

  assert_eq!(run1.len(), run2.len(), "chunk counts must match");
  for (i, (a, b)) in run1.iter().zip(run2.iter()).enumerate() {
    assert!(
      (a - b).abs() < 1e-6,
      "chunk {i} diverged: {a} vs {b} — reset should be bit-exact"
    );
  }
}

/// On a 23-second tram-passing recording (pure environment sound,
/// zero speech), the full pipeline should produce zero segments.
/// This mirrors the audio-bench 3-way comparison result.
#[test]
fn speech_timestamps_on_tram_is_empty() {
  if skip_if_missing_model() {
    return;
  }
  let Some(audio) = read_wav_16k_mono(&tram_path()) else {
    return;
  };
  let segments =
    speech_timestamps(model_path(), &audio, 16_000, VadConfig::default()).expect("run vad");
  println!(
    "tram: {} samples / {:.1}s → {} segments",
    audio.len(),
    audio.len() as f64 / 16_000.0,
    segments.len()
  );
  assert!(
    segments.is_empty(),
    "tram audio should yield zero speech segments, got {}",
    segments.len()
  );
}

/// On the 12-minute balanced dialog, we should detect a substantial
/// number of speech segments and the total speech time should be in
/// a plausible range (~20–50 % of audio length). The audio-bench
/// 3-way benchmark reports 31.2 % speech on this file; allow a wide
/// tolerance here because the upstream tool uses slightly different
/// min-silence defaults.
#[test]
fn speech_timestamps_on_dialog_is_plausible() {
  if skip_if_missing_model() {
    return;
  }
  let Some(audio) = read_wav_16k_mono(&dialog_path()) else {
    return;
  };

  let segments =
    speech_timestamps(model_path(), &audio, 16_000, VadConfig::default()).expect("run vad");

  let total_samples: u64 = segments.iter().map(|s| s.sample_count()).sum();
  let total_speech_s = total_samples as f64 / 16_000.0;
  let audio_s = audio.len() as f64 / 16_000.0;
  let ratio = total_speech_s / audio_s;

  println!(
    "dialog: {:.1}s audio → {} segments, {:.1}s speech ({:.1}%)",
    audio_s,
    segments.len(),
    total_speech_s,
    ratio * 100.0
  );

  assert!(
    segments.len() > 20,
    "dialog should yield >20 speech segments, got {}",
    segments.len()
  );
  assert!(
    (0.15..0.60).contains(&ratio),
    "dialog speech ratio should be in [15%, 60%], got {:.1}%",
    ratio * 100.0
  );

  // Every segment must be a valid half-open interval within the
  // audio bounds and sorted by start.
  let audio_len = audio.len() as u64;
  let mut last_end = 0_u64;
  for s in &segments {
    assert!(
      s.start >= last_end,
      "segments must be sorted non-overlapping"
    );
    assert!(s.end > s.start, "segment must have positive duration");
    assert!(s.end <= audio_len, "segment end beyond audio");
    last_end = s.end;
  }
}

/// A pre-loaded model can be reused across multiple
/// `speech_timestamps_with_model` calls without state leakage.
#[test]
fn speech_timestamps_with_model_is_independent_across_calls() {
  if skip_if_missing_model() {
    return;
  }
  let Some(tram) = read_wav_16k_mono(&tram_path()) else {
    return;
  };
  let Some(dialog) = read_wav_16k_mono(&dialog_path()) else {
    return;
  };

  let mut model = SileroModel::from_path(model_path()).expect("model loads");

  // Interleave tram / dialog / tram. The tram runs must both return
  // zero segments regardless of what was processed between them.
  let tram1 =
    speech_timestamps_with_model(&mut model, &tram, 16_000, VadConfig::default()).unwrap();
  let dialog_segs = speech_timestamps_with_model(
    &mut model,
    &dialog[..60 * 16_000],
    16_000,
    VadConfig::default(),
  )
  .unwrap();
  let tram2 =
    speech_timestamps_with_model(&mut model, &tram, 16_000, VadConfig::default()).unwrap();

  println!(
    "tram1={}, dialog(60s)={}, tram2={}",
    tram1.len(),
    dialog_segs.len(),
    tram2.len()
  );

  assert!(tram1.is_empty(), "first tram run should be empty");
  assert!(tram2.is_empty(), "second tram run should be empty");
  assert!(
    dialog_segs.len() > 3,
    "60 s of dialog should yield >3 segments"
  );
}

/// `VadIterator` streaming over the dialog produces a plausible event
/// stream: some Start events and matching End events, with Start
/// preceding End for each region.
#[test]
fn vad_iterator_emits_balanced_events_on_dialog() {
  if skip_if_missing_model() {
    return;
  }
  let Some(audio) = read_wav_16k_mono(&dialog_path()) else {
    return;
  };

  let mut vad = VadIterator::new(model_path(), VadConfig::default()).expect("load");
  let mut starts = 0_u32;
  let mut ends = 0_u32;
  let mut last_kind = None::<&'static str>; // "start" or "end"

  for chunk in audio.chunks_exact(512) {
    if let Some(event) = vad.process(chunk).expect("process") {
      match event {
        silero::Event::Start(_) => {
          // Can't have two starts in a row without an end.
          assert_ne!(last_kind, Some("start"));
          starts += 1;
          last_kind = Some("start");
        }
        silero::Event::End(_) => {
          // Can't have an end without a preceding start.
          assert_eq!(last_kind, Some("start"));
          ends += 1;
          last_kind = Some("end");
        }
      }
    }
  }

  if let Some(silero::Event::End(_)) = vad.finish() {
    ends += 1;
  }

  println!("streaming events: {starts} starts, {ends} ends");
  assert!(
    starts >= 5,
    "expected at least 5 speech onsets, got {starts}"
  );
  assert_eq!(starts, ends, "every start must be matched by an end");
}

/// `VadIterator::finish` flushes a mid-stream-truncated speech region
/// as a synthetic `End` event.
#[test]
fn vad_iterator_finish_flushes_open_region() {
  if skip_if_missing_model() {
    return;
  }
  let Some(audio) = read_wav_16k_mono(&dialog_path()) else {
    return;
  };

  let mut vad = VadIterator::new(model_path(), VadConfig::default()).expect("load");
  // Drive 5 seconds of audio then stop. If the last 5 s ends mid-speech,
  // `finish` should emit an End event; otherwise it returns None.
  for chunk in audio[..5 * 16_000].chunks_exact(512) {
    let _ = vad.process(chunk).unwrap();
  }
  let tail = vad.finish();
  if vad.is_triggered() {
    // `finish` also resets, so `is_triggered` should now be false.
    panic!("finish should have cleared the triggered state");
  }
  match tail {
    None => println!("no open region at cutoff (fine)"),
    Some(silero::Event::End(sample)) => {
      println!("flushed end at sample {sample}");
      assert!(sample > 0, "flushed end must be positive");
    }
    Some(other) => panic!("unexpected event from finish: {other:?}"),
  }
}
