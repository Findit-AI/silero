//! Optional WAV file I/O helpers.
//!
//! Enabled by the `wav` Cargo feature. Provides a small, opinionated
//! `read_wav` that decodes a WAV file into mono f32 samples at its
//! native sample rate — enough for the `silero` crate's own
//! integration tests and for quick user scripts, but **not** a
//! general-purpose audio library. If you need anything more
//! sophisticated (non-PCM codecs, multi-channel downmixing, format
//! negotiation), use a proper crate like [`symphonia`] or decode
//! upstream in your application.

use std::path::Path;

use crate::error::{Result, SileroError};

/// Read a WAV file into `(samples, sample_rate)`.
///
/// Supports both integer PCM (8 / 16 / 24 / 32 bit) and IEEE float
/// (32-bit) WAV files. Multi-channel inputs are downmixed to mono by
/// averaging channels frame-by-frame — acceptable for speech but not
/// an audiophile-grade conversion.
///
/// # Errors
///
/// Returns [`SileroError::InvalidInput`] if the file cannot be
/// opened, is not a valid WAV container, or contains an unsupported
/// sample format.
pub fn read_wav(path: impl AsRef<Path>) -> Result<(Vec<f32>, u32)> {
  let path = path.as_ref();
  let reader = hound::WavReader::open(path).map_err(|_| {
    SileroError::InvalidInput("failed to open WAV file (check that the path is a valid WAV)")
  })?;
  let spec = reader.spec();
  let channels = spec.channels as usize;
  if channels == 0 {
    return Err(SileroError::InvalidInput("WAV file reports 0 channels"));
  }

  let samples: Vec<f32> = match spec.sample_format {
    hound::SampleFormat::Int => {
      let max = (1_i64 << (spec.bits_per_sample - 1)) as f32;
      reader
        .into_samples::<i32>()
        .map(|s| s.map(|v| v as f32 / max))
        .collect::<std::result::Result<_, _>>()
        .map_err(|_| SileroError::InvalidInput("failed to decode integer PCM samples"))?
    }
    hound::SampleFormat::Float => reader
      .into_samples::<f32>()
      .collect::<std::result::Result<_, _>>()
      .map_err(|_| SileroError::InvalidInput("failed to decode float PCM samples"))?,
  };

  let mono: Vec<f32> = if channels == 1 {
    samples
  } else {
    samples
      .chunks(channels)
      .map(|frame| frame.iter().sum::<f32>() / channels as f32)
      .collect()
  };

  Ok((mono, spec.sample_rate))
}
