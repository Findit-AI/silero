pub use ort::session::builder::GraphOptimizationLevel;

use crate::error::{Error, Result};

/// Sample rates directly supported by the Silero VAD model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SampleRate {
  /// 8 kHz sample rate, which uses smaller chunks and less context.
  Rate8k,
  /// 16 kHz sample rate, which uses larger chunks and more context for better accuracy.
  Rate16k,
}

impl SampleRate {
  /// Returns the sample rate in Hz.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn hz(self) -> u32 {
    match self {
      Self::Rate8k => 8_000,
      Self::Rate16k => 16_000,
    }
  }

  /// Returns the number of samples in a single model chunk for this sample rate.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn chunk_samples(self) -> usize {
    match self {
      Self::Rate8k => 256,
      Self::Rate16k => 512,
    }
  }

  /// Returns the number of context samples the model expects for this sample rate.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn context_samples(self) -> usize {
    match self {
      Self::Rate8k => 32,
      Self::Rate16k => 64,
    }
  }

  /// Create a `SampleRate` from a raw Hz value, returning an error if the rate is not supported by the model.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn from_hz(rate: u32) -> Result<Self> {
    match rate {
      8_000 => Ok(Self::Rate8k),
      16_000 => Ok(Self::Rate16k),
      other => Err(Error::UnsupportedSampleRate { rate: other }),
    }
  }
}

impl Default for SampleRate {
  #[inline]
  fn default() -> Self {
    Self::Rate16k
  }
}

/// Options for constructing an ONNX session.
///
/// This type intentionally stays small. Deployment-specific runtime
/// policy such as `intra_threads` / `inter_threads` should normally be
/// configured one layer up, then passed down via
/// [`crate::Session::from_ort_session`].
#[derive(Debug, Clone, Copy)]
pub struct SessionOptions {
  optimization_level: GraphOptimizationLevel,
}

impl Default for SessionOptions {
  #[inline]
  fn default() -> Self {
    Self::new()
  }
}

impl SessionOptions {
  /// Create a new `SessionOptions` with default values.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn new() -> Self {
    Self {
      optimization_level: GraphOptimizationLevel::Disable,
    }
  }

  /// Returns the graph optimization level to use when constructing the ONNX session.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn optimization_level(&self) -> GraphOptimizationLevel {
    self.optimization_level
  }

  /// Set the graph optimization level to use when constructing the ONNX session.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_optimization_level(mut self, level: GraphOptimizationLevel) -> Self {
    self.optimization_level = level;
    self
  }
}

/// Configuration for turning frame probabilities into speech segments.
#[derive(Debug, Clone, Copy)]
pub struct SpeechOptions {
  sample_rate: SampleRate,
  start_threshold: f32,
  end_threshold: Option<f32>,
  min_speech_duration_ms: u32,
  min_silence_duration_ms: u32,
  speech_pad_ms: u32,
}

impl Default for SpeechOptions {
  fn default() -> Self {
    Self::new()
  }
}

impl SpeechOptions {
  /// Create a new `SpeechOptions` with default values.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn new() -> Self {
    Self {
      sample_rate: SampleRate::Rate16k,
      start_threshold: 0.5,
      end_threshold: None,
      min_speech_duration_ms: 250,
      min_silence_duration_ms: 100,
      speech_pad_ms: 30,
    }
  }

  /// Returns the sample rate to use for speech detection.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn sample_rate(&self) -> SampleRate {
    self.sample_rate
  }

  /// Returns the start threshold, which is the minimum probability required to consider a frame as speech.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn start_threshold(&self) -> f32 {
    self.start_threshold
  }

  /// Returns the end threshold, which is either the explicitly set value or a default derived from the start threshold.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn end_threshold(&self) -> f32 {
    self
      .end_threshold
      .unwrap_or_else(|| sanitize_probability((self.start_threshold - 0.15).max(0.01)))
  }

  /// Returns the minimum duration of detected speech segments, in milliseconds.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn min_speech_duration_ms(&self) -> u32 {
    self.min_speech_duration_ms
  }

  /// Returns the minimum duration of silence required to close a detected speech segment, in milliseconds.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn min_silence_duration_ms(&self) -> u32 {
    self.min_silence_duration_ms
  }

  /// Returns the amount of padding to add to the start of detected speech segments, in milliseconds.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn speech_pad_ms(&self) -> u32 {
    self.speech_pad_ms
  }

  /// Returns the minimum duration of detected speech segments, in samples.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn min_speech_samples(&self) -> u64 {
    ms_to_samples(self.min_speech_duration_ms, self.sample_rate)
  }

  /// Returns the minimum duration of silence required to close a detected speech segment, in samples.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn min_silence_samples(&self) -> u64 {
    ms_to_samples(self.min_silence_duration_ms, self.sample_rate)
  }

  /// Returns the amount of padding to add to the start of detected speech segments, in samples.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn speech_pad_samples(&self) -> u64 {
    ms_to_samples(self.speech_pad_ms, self.sample_rate)
  }

  /// Set the sample rate to use for speech detection.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_sample_rate(mut self, sample_rate: SampleRate) -> Self {
    self.sample_rate = sample_rate;
    self
  }

  /// Set the start threshold, which must be between 0 and 1. If not set, it defaults to 0.5.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_start_threshold(mut self, threshold: f32) -> Self {
    self.start_threshold = sanitize_probability(threshold);
    self
  }

  /// Set the end threshold, which must be less than the start threshold. If not set, it will be automatically derived from the start threshold with a fixed offset.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_end_threshold(mut self, threshold: f32) -> Self {
    self.end_threshold = Some(sanitize_probability(threshold));
    self
  }

  /// Clear the end threshold, causing it to be automatically derived from the start threshold with a fixed offset.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn clear_end_threshold(mut self) -> Self {
    self.end_threshold = None;
    self
  }

  /// Set the minimum duration of detected speech segments, in milliseconds.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_min_speech_duration_ms(mut self, duration_ms: u32) -> Self {
    self.min_speech_duration_ms = duration_ms;
    self
  }

  /// Set the minimum duration of silence required to close a detected speech segment, in milliseconds.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_min_silence_duration_ms(mut self, duration_ms: u32) -> Self {
    self.min_silence_duration_ms = duration_ms;
    self
  }

  /// Set the amount of padding to add to the start of detected speech segments, in milliseconds.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_speech_pad_ms(mut self, pad_ms: u32) -> Self {
    self.speech_pad_ms = pad_ms;
    self
  }
}

#[inline]
pub(crate) fn ms_to_samples(duration_ms: u32, sample_rate: SampleRate) -> u64 {
  (u64::from(duration_ms) * u64::from(sample_rate.hz())) / 1_000
}

#[inline]
fn sanitize_probability(value: f32) -> f32 {
  if value.is_finite() {
    value.clamp(0.0, 1.0)
  } else {
    0.0
  }
}

#[cfg(test)]
mod tests {
  use ort::session::builder::GraphOptimizationLevel;

  use super::{SampleRate, SessionOptions, SpeechOptions, ms_to_samples};

  #[test]
  fn sample_rate_contract_matches_silero_model() {
    assert_eq!(SampleRate::Rate16k.chunk_samples(), 512);
    assert_eq!(SampleRate::Rate16k.context_samples(), 64);
    assert_eq!(SampleRate::Rate8k.chunk_samples(), 256);
    assert_eq!(SampleRate::Rate8k.context_samples(), 32);
  }

  #[test]
  fn speech_config_defaults_match_expected_streaming_behavior() {
    let config = SpeechOptions::default();
    assert_eq!(config.sample_rate(), SampleRate::Rate16k);
    assert_eq!(config.start_threshold(), 0.5);
    assert_eq!(config.end_threshold(), 0.35);
    assert_eq!(config.min_speech_duration_ms(), 250);
    assert_eq!(config.min_silence_duration_ms(), 100);
    assert_eq!(config.speech_pad_ms(), 30);
  }

  #[test]
  fn ms_to_samples_uses_stream_rate() {
    assert_eq!(ms_to_samples(100, SampleRate::Rate16k), 1_600);
    assert_eq!(ms_to_samples(100, SampleRate::Rate8k), 800);
  }

  #[test]
  fn session_options_default_to_unopinionated_core_settings() {
    let options = SessionOptions::default();
    assert_eq!(
      options.optimization_level(),
      GraphOptimizationLevel::Disable
    );
  }
}
