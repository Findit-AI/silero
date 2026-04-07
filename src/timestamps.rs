//! Batch speech-timestamp extraction — the Rust port of the Python
//! `get_speech_timestamps` function.
//!
//! This is the high-level entry point most users will want. Given a
//! whole mono f32 audio buffer and a [`VadConfig`], it returns a list
//! of [`SpeechSegment`]s with padded `start`/`end` sample indices.
//!
//! # Fidelity to upstream
//!
//! The algorithm is a line-by-line port of
//! `silero_vad.utils_vad.get_speech_timestamps` from
//! <https://github.com/snakers4/silero-vad>, including:
//!
//! - Zero-padding the final chunk if it's shorter than `chunk_size`.
//! - The full state machine with `temp_end` / `prev_end` /
//!   `next_start` tracking.
//! - `max_speech_duration_ms` splitting via the longest internal
//!   silence (`use_max_poss_sil_at_max_speech=True` upstream default).
//! - Asymmetric padding at segment boundaries, including the
//!   "silence between two segments is smaller than `2 * speech_pad`,
//!   so split it in half" rule.
//! - End-of-stream flush of a still-open speech region, gated on
//!   `min_speech_samples`.
//! - The min-sample-count gate applied when finalizing a segment
//!   inside the main loop.
//!
//! Anything the upstream does, this port does identically. Anything
//! new — automatic sample-rate downsampling, error handling, the
//! caller-visible [`SpeechSegment`] type — is additive.

use std::path::Path;

use crate::config::{SampleRate, VadConfig};
use crate::error::{Result, SileroError};
use crate::model::SileroModel;

/// A contiguous speech region in the input audio.
///
/// Sample indices are absolute within the input buffer (starting at
/// zero) and already include the configured [`VadConfig`] padding
/// (`speech_pad_ms`). They follow Rust's half-open interval
/// convention: the region covers samples `[start, end)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpeechSegment {
  /// Inclusive start sample index.
  pub start: u64,
  /// Exclusive end sample index.
  pub end: u64,
}

impl SpeechSegment {
  /// Number of samples in the region.
  #[inline]
  pub const fn sample_count(self) -> u64 {
    self.end - self.start
  }

  /// Start time in seconds at the given sample rate.
  #[inline]
  pub fn start_seconds(self, rate: SampleRate) -> f64 {
    self.start as f64 / rate.hz() as f64
  }

  /// End time in seconds at the given sample rate.
  #[inline]
  pub fn end_seconds(self, rate: SampleRate) -> f64 {
    self.end as f64 / rate.hz() as f64
  }

  /// Duration in seconds at the given sample rate.
  #[inline]
  pub fn duration_seconds(self, rate: SampleRate) -> f64 {
    self.sample_count() as f64 / rate.hz() as f64
  }
}

/// Run VAD on a whole audio buffer and return every detected speech
/// segment.
///
/// Loads the ONNX model from disk, runs inference chunk-by-chunk
/// over the input, applies the upstream state machine, and returns
/// the post-processed segments with padding applied. Equivalent to
/// the Python `get_speech_timestamps(audio, model, **config)` call,
/// except it accepts a path rather than a pre-loaded model.
///
/// For repeated calls on many audio files, use
/// [`speech_timestamps_with_model`] to avoid loading the model more
/// than once.
///
/// # Arguments
///
/// - `model_path`: filesystem path to the `silero_vad.onnx` bundle.
/// - `audio`: mono f32 PCM samples at the sample rate declared in
///   `config`. Values outside `[-1, 1]` will not crash but may
///   produce unreliable probabilities; feed normalised audio if
///   possible.
/// - `rate`: the sample rate of `audio`. Must be 8 kHz, 16 kHz, or
///   an integer multiple of 16 kHz (e.g. 32 kHz, 48 kHz, 96 kHz),
///   which are downsampled to 16 kHz automatically by naive
///   decimation, matching upstream Python. Everything else must be
///   resampled by the caller.
/// - `config`: VAD tuning parameters. The `sample_rate` field of
///   `config` is overwritten with the resolved (possibly
///   downsampled) rate inside this function.
pub fn speech_timestamps(
  model_path: impl AsRef<Path>,
  audio: &[f32],
  rate: u32,
  config: VadConfig,
) -> Result<Vec<SpeechSegment>> {
  let mut model = SileroModel::from_path(model_path)?;
  speech_timestamps_with_model(&mut model, audio, rate, config)
}

/// Run VAD with a pre-loaded model. See [`speech_timestamps`].
///
/// The model's sample rate and streaming state are reset at the
/// beginning of this call so consecutive invocations are independent.
pub fn speech_timestamps_with_model(
  model: &mut SileroModel,
  audio: &[f32],
  rate: u32,
  mut config: VadConfig,
) -> Result<Vec<SpeechSegment>> {
  // ── 1. Resolve sample rate, with automatic decimation for
  //       integer-multiples of 16 kHz (upstream Python semantics).
  let (effective_rate, step, audio_owned);
  let audio_slice: &[f32] = if rate == 8_000 || rate == 16_000 {
    effective_rate = SampleRate::from_hz(rate)?;
    step = 1_usize;
    audio
  } else if rate > 16_000 && rate % 16_000 == 0 {
    effective_rate = SampleRate::Rate16k;
    step = (rate / 16_000) as usize;
    audio_owned = audio.iter().step_by(step).copied().collect::<Vec<f32>>();
    &audio_owned[..]
  } else {
    return Err(SileroError::UnsupportedSampleRate { rate });
  };
  // `audio_owned` is unused in the direct-rate branches; silence the
  // compiler warning by binding it only when needed.
  let _ = step;

  config = config.with_sample_rate(effective_rate);
  model.set_sample_rate(effective_rate);
  model.reset();

  // ── 2. Derived constants, mirroring the Python variable names.
  let chunk_size = effective_rate.chunk_samples();
  let threshold = config.threshold();
  let neg_threshold = config.effective_negative_threshold();
  let min_speech_samples = config.min_speech_samples();
  let min_silence_samples = config.min_silence_samples();
  let speech_pad_samples = config.speech_pad_samples();
  let max_speech_samples = config.max_speech_samples();
  let min_silence_at_max_speech_samples = config.min_silence_at_max_speech_samples();
  let audio_len = audio_slice.len() as u64;

  // ── 3. Run inference over every chunk, zero-padding the last one
  //       if it's short. This matches the upstream loop structure.
  let mut probs: Vec<f32> = Vec::with_capacity((audio_slice.len() / chunk_size) + 1);
  let mut pad_buf = vec![0.0_f32; chunk_size];
  let mut offset = 0usize;
  while offset < audio_slice.len() {
    let end = (offset + chunk_size).min(audio_slice.len());
    let real_len = end - offset;
    let prob = if real_len == chunk_size {
      model.process(&audio_slice[offset..end])?
    } else {
      // Pad the final partial chunk with zeros. Upstream Python does
      // `torch.nn.functional.pad(chunk, (0, chunk_size - len))`.
      pad_buf[..real_len].copy_from_slice(&audio_slice[offset..end]);
      pad_buf[real_len..].fill(0.0);
      model.process(&pad_buf)?
    };
    probs.push(prob);
    offset += chunk_size;
  }

  // ── 4. State machine. Names mirror the upstream Python
  //       (`triggered`, `current_speech`, `temp_end`, `prev_end`,
  //       `next_start`, `possible_ends`) so the port stays easy to
  //       verify side by side with the reference.
  #[derive(Debug, Clone, Copy)]
  struct Open {
    start: u64,
  }
  let mut speeches: Vec<SpeechSegment> = Vec::new();

  let mut triggered = false;
  let mut current: Option<Open> = None;
  let mut temp_end: u64 = 0;
  let mut prev_end: u64 = 0;
  let mut next_start: u64 = 0;
  // `(temp_end, silence_duration)` candidates for the longest-silence
  // split used inside `max_speech_samples` overflow handling.
  let mut possible_ends: Vec<(u64, u64)> = Vec::new();

  for (i, &prob) in probs.iter().enumerate() {
    let cur_sample = (i as u64) * (chunk_size as u64);

    // Condition 1: speech returns after tentative silence.
    if prob >= threshold && temp_end > 0 {
      let sil_dur = cur_sample.saturating_sub(temp_end);
      if sil_dur > min_silence_at_max_speech_samples {
        possible_ends.push((temp_end, sil_dur));
      }
      temp_end = 0;
      if next_start < prev_end {
        next_start = cur_sample;
      }
    }

    // Condition 2: speech onset.
    if prob >= threshold && !triggered {
      triggered = true;
      current = Some(Open { start: cur_sample });
      continue;
    }

    // Condition 3: maximum speech duration exceeded.
    if triggered && current.is_some() && max_speech_samples != u64::MAX {
      let open = current.unwrap();
      let duration = cur_sample.saturating_sub(open.start);
      if duration > max_speech_samples {
        // Prefer the longest internal silence as the split point if
        // we recorded any. Otherwise fall back to aggressive cut at
        // the previous silence (`prev_end`) or the current sample.
        if !possible_ends.is_empty() {
          let (split_end, split_dur) = possible_ends
            .iter()
            .copied()
            .max_by_key(|(_, d)| *d)
            .unwrap();
          speeches.push(SpeechSegment {
            start: open.start,
            end: split_end,
          });
          // Prepare next segment if the silence region has an
          // exploitable tail.
          next_start = split_end + split_dur;
          if next_start < split_end + cur_sample {
            current = Some(Open { start: next_start });
          } else {
            triggered = false;
            current = None;
          }
          prev_end = 0;
          next_start = 0;
          temp_end = 0;
          possible_ends.clear();
        } else if prev_end > 0 {
          speeches.push(SpeechSegment {
            start: open.start,
            end: prev_end,
          });
          if next_start < prev_end {
            triggered = false;
            current = None;
          } else {
            current = Some(Open { start: next_start });
          }
          prev_end = 0;
          next_start = 0;
          temp_end = 0;
          possible_ends.clear();
        } else {
          // No previous silence available — cut aggressively at the
          // current sample.
          speeches.push(SpeechSegment {
            start: open.start,
            end: cur_sample,
          });
          triggered = false;
          current = None;
          prev_end = 0;
          next_start = 0;
          temp_end = 0;
          possible_ends.clear();
        }
      }
    }

    // Condition 4: silence detected while inside a speech region.
    if prob < neg_threshold && triggered {
      if temp_end == 0 {
        temp_end = cur_sample;
      }
      let sil_dur_now = cur_sample.saturating_sub(temp_end);
      if sil_dur_now < min_silence_samples {
        continue;
      }
      // Silence is long enough to end this segment.
      if let Some(open) = current.take() {
        if temp_end > open.start && (temp_end - open.start) > min_speech_samples {
          speeches.push(SpeechSegment {
            start: open.start,
            end: temp_end,
          });
        }
      }
      prev_end = 0;
      next_start = 0;
      temp_end = 0;
      triggered = false;
      possible_ends.clear();
    }
  }

  // ── 5. End-of-stream: flush any still-open region if it's long
  //       enough. Matches the upstream unconditional tail flush.
  if let Some(open) = current {
    if audio_len > open.start && (audio_len - open.start) > min_speech_samples {
      speeches.push(SpeechSegment {
        start: open.start,
        end: audio_len,
      });
    }
  }

  // ── 6. Apply padding with the upstream's asymmetric rule.
  apply_padding(&mut speeches, speech_pad_samples, audio_len);

  Ok(speeches)
}

/// Apply `speech_pad_samples` of context padding to each segment,
/// following the upstream Python rules:
///
/// - The first segment's start is shifted back by at most `pad`,
///   clamped to 0.
/// - The last segment's end is shifted forward by at most `pad`,
///   clamped to `audio_len`.
/// - For each pair of adjacent segments, if the silence between them
///   is at least `2 * pad`, both sides get their full padding.
///   Otherwise the silence is split in half and shared.
fn apply_padding(speeches: &mut [SpeechSegment], pad: u64, audio_len: u64) {
  if speeches.is_empty() {
    return;
  }
  let n = speeches.len();

  for i in 0..n {
    if i == 0 {
      speeches[i].start = speeches[i].start.saturating_sub(pad);
    }

    if i != n - 1 {
      let next_start = speeches[i + 1].start;
      let silence = next_start.saturating_sub(speeches[i].end);
      if silence < 2 * pad {
        // Not enough silence for full pad on both sides → split.
        let half = silence / 2;
        speeches[i].end += half;
        // Clamp start to 0 just like upstream.
        speeches[i + 1].start = speeches[i + 1].start.saturating_sub(half);
      } else {
        speeches[i].end = (speeches[i].end + pad).min(audio_len);
        speeches[i + 1].start = speeches[i + 1].start.saturating_sub(pad);
      }
    } else {
      speeches[i].end = (speeches[i].end + pad).min(audio_len);
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  fn seg(s: u64, e: u64) -> SpeechSegment {
    SpeechSegment { start: s, end: e }
  }

  #[test]
  fn speech_segment_geometry() {
    let s = seg(1600, 4800);
    assert_eq!(s.sample_count(), 3200);
    assert!((s.start_seconds(SampleRate::Rate16k) - 0.1).abs() < 1e-9);
    assert!((s.end_seconds(SampleRate::Rate16k) - 0.3).abs() < 1e-9);
    assert!((s.duration_seconds(SampleRate::Rate16k) - 0.2).abs() < 1e-9);
  }

  #[test]
  fn apply_padding_empty_is_noop() {
    let mut segs: Vec<SpeechSegment> = Vec::new();
    apply_padding(&mut segs, 480, 16_000);
    assert!(segs.is_empty());
  }

  #[test]
  fn apply_padding_single_segment_clamps_to_audio_bounds() {
    let mut segs = vec![seg(100, 1500)];
    apply_padding(&mut segs, 480, 2000);
    // Start clamped to 0 (100 - 480 would be negative).
    assert_eq!(segs[0].start, 0);
    // End clamped to audio_len (1500 + 480 = 1980 < 2000 so no clamp).
    assert_eq!(segs[0].end, 1980);
  }

  #[test]
  fn apply_padding_end_clamps_to_audio_len() {
    let mut segs = vec![seg(0, 1500)];
    apply_padding(&mut segs, 480, 1700);
    assert_eq!(segs[0].start, 0);
    // 1500 + 480 = 1980 > 1700 → clamped.
    assert_eq!(segs[0].end, 1700);
  }

  #[test]
  fn apply_padding_two_segments_with_sufficient_silence() {
    // silence = 2000 - 1500 = 500 ≥ 2*200 (400), each side gets full 200.
    let mut segs = vec![seg(100, 1500), seg(2000, 3000)];
    apply_padding(&mut segs, 200, 5000);
    assert_eq!(segs[0].start, 0); // 100 - 200 clamped
    assert_eq!(segs[0].end, 1700); // 1500 + 200
    assert_eq!(segs[1].start, 1800); // 2000 - 200
    assert_eq!(segs[1].end, 3200); // 3000 + 200
  }

  #[test]
  fn apply_padding_two_segments_with_insufficient_silence_split_half() {
    // silence = 1700 - 1500 = 200, less than 2*150 = 300. Split in half = 100.
    let mut segs = vec![seg(500, 1500), seg(1700, 3000)];
    apply_padding(&mut segs, 150, 5000);
    // First segment: start pad applied (clamped start to 350),
    //                end gets half of silence = 100.
    assert_eq!(segs[0].start, 350); // 500 - 150
    assert_eq!(segs[0].end, 1600); // 1500 + 100
                                   // Second segment: start shifted back by 100, end gets full pad.
    assert_eq!(segs[1].start, 1600); // 1700 - 100
    assert_eq!(segs[1].end, 3150); // 3000 + 150
  }

  #[test]
  fn apply_padding_three_segments_mixed() {
    // Seg 1 (0, 1000) — seg 2 (2000, 3000) — seg 3 (5000, 6000), audio 10_000.
    // Pad = 200.
    // Silence 1→2 = 1000, ≥ 400, full pad both sides.
    // Silence 2→3 = 2000, ≥ 400, full pad both sides.
    let mut segs = vec![seg(0, 1000), seg(2000, 3000), seg(5000, 6000)];
    apply_padding(&mut segs, 200, 10_000);
    assert_eq!(segs[0], seg(0, 1200));
    assert_eq!(segs[1], seg(1800, 3200));
    assert_eq!(segs[2], seg(4800, 6200));
  }

  #[test]
  fn apply_padding_tight_boundary_split() {
    // Adjacent segments touching: silence = 0.
    let mut segs = vec![seg(1000, 2000), seg(2000, 3000)];
    apply_padding(&mut segs, 100, 5000);
    // Silence = 0 < 2*100 = 200, split half = 0 each.
    assert_eq!(segs[0].start, 900);
    assert_eq!(segs[0].end, 2000); // +0
    assert_eq!(segs[1].start, 2000); // -0
    assert_eq!(segs[1].end, 3100);
  }
}
