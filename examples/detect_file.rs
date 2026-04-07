use std::{env, path::PathBuf};

use silero::{Session, SpeechOptions, detect_speech};

const MODEL_BYTES: &[u8] = include_bytes!(concat!(
  env!("CARGO_MANIFEST_DIR"),
  "/models/silero_vad.onnx"
));

fn main() -> Result<(), Box<dyn std::error::Error>> {
  let wav_path: PathBuf = env::args()
    .nth(1)
    .ok_or("usage: detect_file <wav_path>")?
    .into();
  let mut reader = hound::WavReader::open(&wav_path)?;
  let spec = reader.spec();
  if spec.channels != 1 {
    return Err(format!("expected mono WAV input, got {} channels", spec.channels).into());
  }

  let sample_rate = silero::SampleRate::from_hz(spec.sample_rate)?;
  let audio: Vec<f32> = match spec.sample_format {
    hound::SampleFormat::Float => reader.samples::<f32>().collect::<Result<Vec<_>, _>>()?,
    hound::SampleFormat::Int => {
      let scale = (1_i64 << (spec.bits_per_sample.saturating_sub(1))) as f32;
      reader
        .samples::<i32>()
        .map(|sample| sample.map(|value| value as f32 / scale))
        .collect::<Result<Vec<_>, _>>()?
    }
  };

  let mut session = Session::from_memory(MODEL_BYTES)?;
  let segments = detect_speech(
    &mut session,
    &audio,
    SpeechOptions::default().with_sample_rate(sample_rate),
  )?;

  println!(
    "{} segments detected in {}",
    segments.len(),
    wav_path.display()
  );
  for (index, segment) in segments.iter().enumerate() {
    println!(
      "{:>3}: {:.2}s -> {:.2}s",
      index + 1,
      segment.start_seconds(),
      segment.end_seconds()
    );
  }

  Ok(())
}
