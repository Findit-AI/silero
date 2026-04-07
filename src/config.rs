//! Configuration types and defaults for the Silero VAD pipeline.
//!
//! Every default value in this module is taken verbatim from the
//! upstream Python reference implementation at
//! <https://github.com/snakers4/silero-vad/blob/master/src/silero_vad/utils_vad.py>
//! (function `get_speech_timestamps`). The Rust port aims to produce
//! bit-identical segment boundaries when fed the same audio with the
//! same parameters.

use crate::error::SileroError;

/// A sample rate supported by the Silero VAD model.
///
/// Silero is a streaming RNN trained at exactly these two rates, so we
/// expose them as a closed enum rather than `u32`. This makes it
/// impossible to accidentally pass e.g. 44100 to the low-level API.
///
/// Higher sample rates that are integer multiples of 16 kHz (22050,
/// 32000, 48000, …) are accepted by [`crate::speech_timestamps`]
/// through automatic naive downsampling (decimation), matching the
/// upstream Python behaviour. For everything else, either resample
/// using the optional `resample` feature or upstream of the library.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SampleRate {
  /// 8 kHz, narrow-band speech (telephony, legacy audio).
  Rate8k,
  /// 16 kHz, the standard rate for modern speech processing. This is
  /// the default and what most callers want.
  Rate16k,
}

impl SampleRate {
  /// Returns the sample rate as an integer number of hertz.
  #[inline]
  pub const fn hz(self) -> u32 {
    match self {
      Self::Rate8k => 8_000,
      Self::Rate16k => 16_000,
    }
  }

  /// Number of audio samples per input chunk to the ONNX model.
  ///
  /// This is a hard constraint of the exported model — larger values
  /// produce LSTM shape errors, smaller values produce unreliable
  /// probabilities. 512 samples at 16 kHz = 32 ms per chunk.
  #[inline]
  pub const fn chunk_samples(self) -> usize {
    match self {
      Self::Rate8k => 256,
      Self::Rate16k => 512,
    }
  }

  /// Number of rolling-context samples prepended to each chunk before
  /// the ONNX call.
  ///
  /// The ONNX input tensor expects `[1, context + chunk]` = `[1, 288]`
  /// at 8 kHz or `[1, 576]` at 16 kHz. Passing just the chunk without
  /// the context triggers an internal "too short" branch in the model
  /// and silently produces near-zero probabilities for every input —
  /// this is the single most important implementation detail to get
  /// right, and it is only documented in the upstream Python source.
  #[inline]
  pub const fn context_samples(self) -> usize {
    match self {
      Self::Rate8k => 32,
      Self::Rate16k => 64,
    }
  }

  /// Try to interpret an arbitrary sample rate as one of the directly
  /// supported rates. Returns `Err` for anything else — including
  /// integer multiples of 16 kHz, which are handled one layer up by
  /// [`crate::speech_timestamps`] via decimation rather than here.
  pub fn from_hz(rate: u32) -> Result<Self, SileroError> {
    match rate {
      8_000 => Ok(Self::Rate8k),
      16_000 => Ok(Self::Rate16k),
      other => Err(SileroError::UnsupportedSampleRate { rate: other }),
    }
  }
}

impl Default for SampleRate {
  #[inline]
  fn default() -> Self {
    Self::Rate16k
  }
}

// ── VAD tuning parameters ──────────────────────────

/// Tuning parameters for [`crate::speech_timestamps`] and
/// [`crate::VadIterator`].
///
/// Every field has a sensible default matched to the upstream Python
/// defaults; override only the ones you care about via the builder
/// methods.
///
/// # Example
///
/// ```rust,ignore
/// use silero::{VadConfig, SampleRate};
/// use std::time::Duration;
///
/// // Stricter speech boundary than the default, for STT pre-filtering.
/// let config = VadConfig::default()
///     .with_threshold(0.5)
///     .with_min_silence(Duration::from_millis(200))
///     .with_speech_pad(Duration::from_millis(50));
/// ```
#[derive(Debug, Clone, Copy)]
pub struct VadConfig {
  sample_rate: SampleRate,
  threshold: f32,
  /// `None` means "computed lazily as `max(threshold - 0.15, 0.01)`",
  /// matching upstream Python behaviour. Users that want a specific
  /// value can pin it with [`Self::with_negative_threshold`].
  negative_threshold: Option<f32>,
  min_speech_duration_ms: u32,
  max_speech_duration_ms: u32,
  min_silence_duration_ms: u32,
  speech_pad_ms: u32,
  /// Minimum silence within a long speech region that counts as a
  /// "possible end" candidate when splitting on
  /// `max_speech_duration_s`. Matches the upstream constant.
  min_silence_at_max_speech_ms: u32,
}

impl Default for VadConfig {
  fn default() -> Self {
    Self {
      sample_rate: SampleRate::Rate16k,
      threshold: 0.5,
      negative_threshold: None,
      min_speech_duration_ms: 250,
      // `u32::MAX` milliseconds ≈ 49.7 days; effectively infinity but
      // avoids storing `f64::INFINITY` in a Copy struct.
      max_speech_duration_ms: u32::MAX,
      min_silence_duration_ms: 100,
      speech_pad_ms: 30,
      min_silence_at_max_speech_ms: 98,
    }
  }
}

impl VadConfig {
  /// Create a new config with default values. Equivalent to
  /// [`Self::default`].
  #[inline]
  pub const fn new() -> Self {
    Self {
      sample_rate: SampleRate::Rate16k,
      threshold: 0.5,
      negative_threshold: None,
      min_speech_duration_ms: 250,
      max_speech_duration_ms: u32::MAX,
      min_silence_duration_ms: 100,
      speech_pad_ms: 30,
      min_silence_at_max_speech_ms: 98,
    }
  }

  /// Set the sample rate. Must match the input audio you'll feed.
  #[inline]
  pub const fn with_sample_rate(mut self, rate: SampleRate) -> Self {
    self.sample_rate = rate;
    self
  }

  /// Set the speech probability threshold. Frames above this value are
  /// classified as voiced. Default `0.5`.
  ///
  /// The upstream guidance is: stay close to `0.5`. Lowering catches
  /// quieter speech but adds false positives, especially on
  /// environmental audio. Raising reduces false positives but misses
  /// soft speech.
  #[inline]
  pub const fn with_threshold(mut self, threshold: f32) -> Self {
    self.threshold = threshold;
    self
  }

  /// Override the negative (silence) threshold.
  ///
  /// By default this is computed lazily as `max(threshold - 0.15,
  /// 0.01)` whenever it's first needed, which matches upstream Python.
  /// The 0.15 offset provides hysteresis: once a speech region starts
  /// we only close it when probability drops well below the speech
  /// threshold, to avoid chattering on borderline frames.
  #[inline]
  pub const fn with_negative_threshold(mut self, neg_threshold: f32) -> Self {
    self.negative_threshold = Some(neg_threshold);
    self
  }

  /// Minimum speech segment duration in milliseconds. Shorter regions
  /// are discarded. Default `250` ms.
  #[inline]
  pub const fn with_min_speech_ms(mut self, ms: u32) -> Self {
    self.min_speech_duration_ms = ms;
    self
  }

  /// Maximum speech segment duration in milliseconds. Segments longer
  /// than this are split at the longest internal silence (or, if no
  /// sufficient silence exists, cut aggressively at the boundary).
  /// Default effectively infinity (no splitting).
  #[inline]
  pub const fn with_max_speech_ms(mut self, ms: u32) -> Self {
    self.max_speech_duration_ms = ms;
    self
  }

  /// Minimum silence gap between two speech regions. Silences shorter
  /// than this are absorbed into the surrounding speech. Default
  /// `100` ms.
  #[inline]
  pub const fn with_min_silence_ms(mut self, ms: u32) -> Self {
    self.min_silence_duration_ms = ms;
    self
  }

  /// Padding added to each speech segment's start and end, in
  /// milliseconds. Default `30` ms — enough context for most STT
  /// systems to decode the first/last word cleanly without eating into
  /// neighbouring segments.
  ///
  /// When the silence between two segments is shorter than `2 *
  /// speech_pad`, the available silence is split fairly (see the
  /// upstream Python reference for the exact split rule).
  #[inline]
  pub const fn with_speech_pad_ms(mut self, ms: u32) -> Self {
    self.speech_pad_ms = ms;
    self
  }

  /// Minimum silence within a long speech region to count as a
  /// splitting candidate when `max_speech_duration_ms` is exceeded.
  /// Default `98` ms, matching upstream. Rarely needs changing.
  #[inline]
  pub const fn with_min_silence_at_max_speech_ms(mut self, ms: u32) -> Self {
    self.min_silence_at_max_speech_ms = ms;
    self
  }

  // ── Read-only accessors ────────────────────────

  /// Returns the configured sample rate.
  #[inline]
  pub const fn sample_rate(&self) -> SampleRate {
    self.sample_rate
  }

  /// Returns the configured speech threshold.
  #[inline]
  pub const fn threshold(&self) -> f32 {
    self.threshold
  }

  /// Returns the effective negative (silence) threshold.
  ///
  /// If the user never set one explicitly, the upstream formula
  /// `max(threshold - 0.15, 0.01)` is applied here. The 0.01 floor
  /// protects against the case where `threshold < 0.15` and the
  /// negative threshold would otherwise go negative.
  #[inline]
  pub fn effective_negative_threshold(&self) -> f32 {
    self
      .negative_threshold
      .unwrap_or_else(|| (self.threshold - 0.15).max(0.01))
  }

  /// Returns the configured minimum speech duration in milliseconds.
  #[inline]
  pub const fn min_speech_ms(&self) -> u32 {
    self.min_speech_duration_ms
  }

  /// Returns the configured maximum speech duration in milliseconds.
  #[inline]
  pub const fn max_speech_ms(&self) -> u32 {
    self.max_speech_duration_ms
  }

  /// Returns the configured minimum silence duration in milliseconds.
  #[inline]
  pub const fn min_silence_ms(&self) -> u32 {
    self.min_silence_duration_ms
  }

  /// Returns the configured speech padding in milliseconds.
  #[inline]
  pub const fn speech_pad_ms(&self) -> u32 {
    self.speech_pad_ms
  }

  /// Returns the minimum silence within a max-speech region.
  #[inline]
  pub const fn min_silence_at_max_speech_ms(&self) -> u32 {
    self.min_silence_at_max_speech_ms
  }

  // ── Sample-space derived quantities ────────────

  /// Returns the minimum speech duration in samples at the current
  /// sample rate.
  #[inline]
  pub fn min_speech_samples(&self) -> u64 {
    ms_to_samples(self.min_speech_duration_ms, self.sample_rate)
  }

  /// Returns the minimum silence duration in samples.
  #[inline]
  pub fn min_silence_samples(&self) -> u64 {
    ms_to_samples(self.min_silence_duration_ms, self.sample_rate)
  }

  /// Returns the speech pad in samples.
  #[inline]
  pub fn speech_pad_samples(&self) -> u64 {
    ms_to_samples(self.speech_pad_ms, self.sample_rate)
  }

  /// Returns the max-speech threshold in samples, adjusted for the
  /// upstream formula:
  ///
  /// ```text
  /// max_samples = sr * max_s - chunk_samples - 2 * pad_samples
  /// ```
  ///
  /// This reflects the fact that `max_speech_duration_s` in upstream
  /// Python is interpreted inclusive of padding, so the raw internal
  /// limit is tighter. Returns `u64::MAX` if the user left
  /// `max_speech_duration_ms` at its default infinity-like value.
  pub fn max_speech_samples(&self) -> u64 {
    if self.max_speech_duration_ms == u32::MAX {
      return u64::MAX;
    }
    let chunk = self.sample_rate.chunk_samples() as u64;
    let pad = self.speech_pad_samples();
    let raw = ms_to_samples(self.max_speech_duration_ms, self.sample_rate);
    raw.saturating_sub(chunk).saturating_sub(2 * pad)
  }

  /// Returns the minimum silence within a max-speech region, in
  /// samples.
  #[inline]
  pub fn min_silence_at_max_speech_samples(&self) -> u64 {
    ms_to_samples(self.min_silence_at_max_speech_ms, self.sample_rate)
  }
}

/// Convert a millisecond count to an integer sample count at the given
/// sample rate, rounding down. Matches upstream Python's integer
/// division behaviour (`sampling_rate * ms / 1000`).
#[inline]
pub(crate) fn ms_to_samples(ms: u32, rate: SampleRate) -> u64 {
  (rate.hz() as u64).saturating_mul(ms as u64) / 1000
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn sample_rate_hz_roundtrip() {
    assert_eq!(SampleRate::from_hz(8000).unwrap(), SampleRate::Rate8k);
    assert_eq!(SampleRate::from_hz(16000).unwrap(), SampleRate::Rate16k);
    assert!(SampleRate::from_hz(44100).is_err());
    assert!(SampleRate::from_hz(22050).is_err());
    assert!(SampleRate::from_hz(0).is_err());
  }

  #[test]
  fn sample_rate_chunk_and_context() {
    assert_eq!(SampleRate::Rate16k.chunk_samples(), 512);
    assert_eq!(SampleRate::Rate16k.context_samples(), 64);
    assert_eq!(SampleRate::Rate8k.chunk_samples(), 256);
    assert_eq!(SampleRate::Rate8k.context_samples(), 32);
  }

  #[test]
  fn sample_rate_default_is_16k() {
    assert_eq!(SampleRate::default(), SampleRate::Rate16k);
  }

  #[test]
  fn vad_config_defaults_match_upstream() {
    let c = VadConfig::default();
    assert_eq!(c.sample_rate(), SampleRate::Rate16k);
    assert_eq!(c.threshold(), 0.5);
    assert_eq!(c.min_speech_ms(), 250);
    assert_eq!(c.min_silence_ms(), 100);
    assert_eq!(c.speech_pad_ms(), 30);
    assert_eq!(c.min_silence_at_max_speech_ms(), 98);
    assert_eq!(c.max_speech_ms(), u32::MAX);
  }

  #[test]
  fn vad_config_builder_chains() {
    let c = VadConfig::new()
      .with_sample_rate(SampleRate::Rate8k)
      .with_threshold(0.6)
      .with_min_speech_ms(300)
      .with_min_silence_ms(200)
      .with_speech_pad_ms(50);
    assert_eq!(c.sample_rate(), SampleRate::Rate8k);
    assert_eq!(c.threshold(), 0.6);
    assert_eq!(c.min_speech_ms(), 300);
    assert_eq!(c.min_silence_ms(), 200);
    assert_eq!(c.speech_pad_ms(), 50);
  }

  #[test]
  fn effective_negative_threshold_matches_upstream_formula() {
    // default: 0.5 - 0.15 = 0.35
    let c = VadConfig::default();
    assert!((c.effective_negative_threshold() - 0.35).abs() < 1e-6);

    // low threshold clamps to the 0.01 floor
    let c = VadConfig::default().with_threshold(0.10);
    assert!((c.effective_negative_threshold() - 0.01).abs() < 1e-6);

    // threshold exactly at the boundary
    let c = VadConfig::default().with_threshold(0.16);
    assert!((c.effective_negative_threshold() - 0.01).abs() < 1e-6);
    let c = VadConfig::default().with_threshold(0.17);
    assert!((c.effective_negative_threshold() - 0.02).abs() < 1e-6);

    // explicit override wins over the formula
    let c = VadConfig::default().with_negative_threshold(0.42);
    assert!((c.effective_negative_threshold() - 0.42).abs() < 1e-6);
  }

  #[test]
  fn ms_to_samples_rounds_down() {
    // 100 ms @ 16 kHz = exactly 1600 samples
    assert_eq!(ms_to_samples(100, SampleRate::Rate16k), 1600);
    // 30 ms @ 16 kHz = 480 samples
    assert_eq!(ms_to_samples(30, SampleRate::Rate16k), 480);
    // 98 ms @ 16 kHz = 1568 samples
    assert_eq!(ms_to_samples(98, SampleRate::Rate16k), 1568);
    // 100 ms @ 8 kHz = 800 samples
    assert_eq!(ms_to_samples(100, SampleRate::Rate8k), 800);
    // 1 ms @ 16 kHz = 16 samples
    assert_eq!(ms_to_samples(1, SampleRate::Rate16k), 16);
  }

  #[test]
  fn max_speech_samples_handles_infinity_sentinel() {
    let c = VadConfig::default();
    assert_eq!(c.max_speech_samples(), u64::MAX);
  }

  #[test]
  fn max_speech_samples_applies_formula() {
    // 10 s max speech @ 16 kHz = 160000 samples raw
    //     minus chunk (512)
    //     minus 2 * speech_pad (2 * 480) = 960
    //     = 158528 samples
    let c = VadConfig::default().with_max_speech_ms(10_000);
    assert_eq!(c.max_speech_samples(), 160_000 - 512 - 960);
  }

  #[test]
  fn derived_sample_counts_at_16k() {
    let c = VadConfig::default(); // min_speech 250, min_silence 100, pad 30
    assert_eq!(c.min_speech_samples(), 4000);
    assert_eq!(c.min_silence_samples(), 1600);
    assert_eq!(c.speech_pad_samples(), 480);
  }
}
