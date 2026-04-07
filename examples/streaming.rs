//! Streaming example: drive a `VadIterator` with 512-sample chunks
//! and print `Start` / `End` events as they arrive.
//!
//! This is the pattern you'd use for live audio (microphone, network
//! stream, etc.) where you want sub-second latency between speech
//! events and your application's reaction.
//!
//! Usage:
//! ```bash
//! RUSTFLAGS="-L /path/to/onnxruntime" \
//!     cargo run --release --example streaming --features wav -- \
//!     path/to/audio.wav
//! ```

use std::path::PathBuf;

use silero::{io::read_wav, Event, SampleRate, VadConfig, VadIterator};

fn main() -> Result<(), Box<dyn std::error::Error>> {
  let mut args = std::env::args().skip(1);
  let wav_path: PathBuf = args.next().ok_or("usage: streaming <wav_path>")?.into();

  let model_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models/silero_vad.onnx");
  if !model_path.exists() {
    return Err(format!("model not found at {}", model_path.display()).into());
  }

  let (audio, sample_rate) = read_wav(&wav_path)?;
  if sample_rate != 16_000 {
    return Err(
      format!(
        "streaming example requires 16 kHz audio, got {sample_rate} Hz \
       (use the `detect_file` example for automatic resampling)"
      )
      .into(),
    );
  }

  let mut vad = VadIterator::new(&model_path, VadConfig::default())?;
  let chunk_size = vad.chunk_samples();
  println!(
    "Streaming {} samples through VadIterator in {}-sample chunks...\n",
    audio.len(),
    chunk_size
  );

  let rate = SampleRate::Rate16k;
  let to_seconds = |sample: u64| sample as f64 / rate.hz() as f64;

  let mut active_start: Option<u64> = None;
  for chunk in audio.chunks_exact(chunk_size) {
    if let Some(event) = vad.process(chunk)? {
      match event {
        Event::Start(sample) => {
          println!("▶ start  @ {:7.2} s", to_seconds(sample));
          active_start = Some(sample);
        }
        Event::End(sample) => {
          let dur = active_start
            .take()
            .map(|s| to_seconds(sample.saturating_sub(s)))
            .unwrap_or(0.0);
          println!(
            "■ end    @ {:7.2} s   (duration {:.2} s)",
            to_seconds(sample),
            dur
          );
        }
      }
    }
  }

  // Flush any still-open region at end of stream.
  if let Some(Event::End(sample)) = vad.finish() {
    println!("■ end    @ {:7.2} s   (flushed at EOF)", to_seconds(sample));
  }

  Ok(())
}
