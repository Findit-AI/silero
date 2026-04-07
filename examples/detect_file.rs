//! Minimal example: load a WAV file and print all detected speech
//! segments using the batch `speech_timestamps` API.
//!
//! Usage:
//! ```bash
//! RUSTFLAGS="-L /path/to/onnxruntime" \
//!     cargo run --release --example detect_file --features wav -- \
//!     path/to/audio.wav
//! ```
//!
//! The WAV file must be mono; 16 kHz and 8 kHz are processed directly,
//! integer multiples of 16 kHz (32, 48, 96 kHz) are automatically
//! decimated to 16 kHz, anything else is rejected.

use std::path::PathBuf;

use silero::{io::read_wav, speech_timestamps, SampleRate, VadConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
  let mut args = std::env::args().skip(1);
  let wav_path: PathBuf = args.next().ok_or("usage: detect_file <wav_path>")?.into();

  // Model path is relative to the crate root. In a real application
  // you would distribute the ONNX file alongside your binary.
  let model_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models/silero_vad.onnx");
  if !model_path.exists() {
    return Err(format!("model not found at {}", model_path.display()).into());
  }

  println!("Loading {}", wav_path.display());
  let (audio, sample_rate) = read_wav(&wav_path)?;
  let duration_s = audio.len() as f64 / sample_rate as f64;
  println!(
    "  {} samples @ {} Hz = {:.1} s",
    audio.len(),
    sample_rate,
    duration_s
  );

  println!("\nRunning Silero VAD...");
  let t = std::time::Instant::now();
  let segments = speech_timestamps(&model_path, &audio, sample_rate, VadConfig::default())?;
  let elapsed = t.elapsed().as_secs_f64();
  let rtf = duration_s / elapsed;
  println!(
    "  {} segments in {:.2} s ({:.0}x real-time)",
    segments.len(),
    elapsed,
    rtf
  );

  // Report speech ratio relative to the effective (post-decimation)
  // sample rate. For 16 kHz or 8 kHz input this is just the file's
  // own rate; for 32/48 kHz it's 16 kHz.
  let effective_rate = if sample_rate == 8_000 {
    SampleRate::Rate8k
  } else {
    SampleRate::Rate16k
  };
  let total_speech_samples: u64 = segments.iter().map(|s| s.sample_count()).sum();
  let total_speech_s = total_speech_samples as f64 / effective_rate.hz() as f64;
  println!(
    "  speech ratio: {:.1}% ({:.1} s of {:.1} s)",
    total_speech_s / duration_s * 100.0,
    total_speech_s,
    duration_s
  );

  println!("\nSegments:");
  for (i, seg) in segments.iter().enumerate() {
    println!(
      "  {:3}  [{:7.2} s  →  {:7.2} s]  ({:.2} s)",
      i + 1,
      seg.start_seconds(effective_rate),
      seg.end_seconds(effective_rate),
      seg.duration_seconds(effective_rate),
    );
  }

  Ok(())
}
