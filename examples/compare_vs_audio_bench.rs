//! Runs the new `silero` crate and reports per-file speech ratios
//! for apples-to-apples comparison against `audio-bench::vad_compare`.
//!
//! This is a development aid, not a public user-facing example. It's
//! how we validate that the Rust port matches the reference
//! implementation bit-for-bit (or close enough).
//!
//! Usage:
//! ```bash
//! RUSTFLAGS="-L /path/to/onnxruntime" \
//!     cargo run --release --example compare_vs_audio_bench --features wav -- \
//!     /path/to/audio/file/or/directory
//! ```

use std::path::PathBuf;

use silero::{io::read_wav, speech_timestamps_with_model, SampleRate, SileroModel, VadConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
  let mut args = std::env::args().skip(1);
  let path: PathBuf = args
    .next()
    .ok_or("usage: compare_vs_audio_bench <path>")?
    .into();

  let model_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models/silero_vad.onnx");
  if !model_path.exists() {
    return Err(format!("model not found at {}", model_path.display()).into());
  }
  let mut model = SileroModel::from_path(&model_path)?;

  let files = collect_wav_files(&path)?;
  println!("Found {} WAV file(s)\n", files.len());
  println!(
    "{:<45} {:>8} {:>8} {:>8} {:>8} {:>8}",
    "file", "dur(s)", "segs", "speech(s)", "ratio%", "RTF"
  );
  println!("{}", "─".repeat(90));

  let mut total_audio_s = 0.0_f64;
  let mut total_speech_s = 0.0_f64;
  let mut total_infer_ms = 0.0_f64;
  let mut total_segs: usize = 0;

  for f in &files {
    let (audio, sr) = read_wav(f)?;
    let audio_s = audio.len() as f64 / sr as f64;

    let t = std::time::Instant::now();
    let segments = speech_timestamps_with_model(&mut model, &audio, sr, VadConfig::default())?;
    let infer_ms = t.elapsed().as_secs_f64() * 1000.0;

    let effective_rate = if sr == 8_000 {
      SampleRate::Rate8k
    } else {
      SampleRate::Rate16k
    };
    let speech_samples: u64 = segments.iter().map(|s| s.sample_count()).sum();
    let speech_s = speech_samples as f64 / effective_rate.hz() as f64;
    let ratio = speech_s / audio_s;
    let rtf = if infer_ms > 0.0 {
      audio_s * 1000.0 / infer_ms
    } else {
      0.0
    };

    let fname = f.file_name().and_then(|s| s.to_str()).unwrap_or("?");
    // Truncate by char count, not byte count, so multi-byte UTF-8
    // characters (e.g. CJK filenames) don't split mid-codepoint.
    let truncated = if fname.chars().count() > 43 {
      let tail: String = fname
        .chars()
        .rev()
        .take(42)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();
      format!("…{tail}")
    } else {
      fname.to_string()
    };
    println!(
      "{:<45} {:>8.1} {:>8} {:>8.1} {:>7.1}% {:>7.0}x",
      truncated,
      audio_s,
      segments.len(),
      speech_s,
      ratio * 100.0,
      rtf
    );

    total_audio_s += audio_s;
    total_speech_s += speech_s;
    total_infer_ms += infer_ms;
    total_segs += segments.len();
  }

  println!("{}", "─".repeat(90));
  let overall_ratio = total_speech_s / total_audio_s.max(1e-9);
  let overall_rtf = if total_infer_ms > 0.0 {
    total_audio_s * 1000.0 / total_infer_ms
  } else {
    0.0
  };
  println!(
    "{:<45} {:>8.1} {:>8} {:>8.1} {:>7.1}% {:>7.0}x",
    "TOTAL",
    total_audio_s,
    total_segs,
    total_speech_s,
    overall_ratio * 100.0,
    overall_rtf
  );

  Ok(())
}

fn collect_wav_files(path: &std::path::Path) -> std::io::Result<Vec<PathBuf>> {
  if path.is_file() {
    return Ok(vec![path.to_path_buf()]);
  }
  let mut files = Vec::new();
  for entry in std::fs::read_dir(path)? {
    let entry = entry?;
    let p = entry.path();
    if p.extension().and_then(|e| e.to_str()) == Some("wav") {
      files.push(p);
    }
  }
  files.sort();
  Ok(files)
}
