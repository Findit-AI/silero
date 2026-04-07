//! Optional high-quality resampling.
//!
//! Enabled by the `resample` Cargo feature. Wraps [`rubato`]'s
//! SincFixedIn resampler in a simple one-shot function so callers can
//! convert arbitrary-rate audio to 16 kHz (or 8 kHz) before feeding
//! it to [`crate::speech_timestamps`].
//!
//! For integer-multiple sample rates (32 kHz, 48 kHz, …),
//! `speech_timestamps` already performs naive decimation internally
//! — you only need this module for non-integer ratios like 44.1 kHz
//! → 16 kHz.

use crate::error::{Result, SileroError};

/// Resample a mono f32 buffer from `from_hz` to `to_hz` using a
/// Blackman-Harris windowed sinc filter.
///
/// For integer-multiple ratios where decimation is sufficient, you
/// can just pass the audio through [`crate::speech_timestamps`]
/// directly — it handles those internally. Use this function only
/// when the ratio is non-integer (e.g. 44.1 kHz → 16 kHz).
///
/// # Errors
///
/// Returns [`SileroError::InvalidInput`] if the rubato parameter
/// validation fails (e.g. a zero sample rate).
pub fn resample(samples: &[f32], from_hz: u32, to_hz: u32) -> Result<Vec<f32>> {
  if from_hz == to_hz {
    return Ok(samples.to_vec());
  }
  if from_hz == 0 || to_hz == 0 {
    return Err(SileroError::InvalidInput("sample rate must be > 0"));
  }

  use rubato::{
    Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction,
  };

  let params = SincInterpolationParameters {
    sinc_len: 256,
    f_cutoff: 0.95,
    oversampling_factor: 256,
    interpolation: SincInterpolationType::Linear,
    window: WindowFunction::BlackmanHarris2,
  };

  let ratio = to_hz as f64 / from_hz as f64;
  let chunk = 1024;
  let mut resampler = SincFixedIn::<f32>::new(ratio, 2.0, params, chunk, 1)
    .map_err(|_| SileroError::InvalidInput("rubato resampler rejected the given sample rates"))?;

  let mut out = Vec::with_capacity((samples.len() as f64 * ratio * 1.1) as usize);
  for chunk_in in samples.chunks(chunk) {
    let padded = if chunk_in.len() < chunk {
      let mut p = chunk_in.to_vec();
      p.resize(chunk, 0.0);
      vec![p]
    } else {
      vec![chunk_in.to_vec()]
    };
    let processed = resampler
      .process(
        &padded.iter().map(|v| v.as_slice()).collect::<Vec<_>>(),
        None,
      )
      .map_err(|_| SileroError::InvalidInput("rubato resampler failed to process a chunk"))?;
    out.extend_from_slice(&processed[0]);
  }

  // Trim the tail to the analytic expected output length, matching
  // how downstream callers typically want the result.
  let expected = (samples.len() as f64 * ratio).round() as usize;
  out.truncate(expected);
  Ok(out)
}
