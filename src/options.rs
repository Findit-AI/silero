use core::time::Duration;

#[cfg(feature = "onnx")]
#[cfg_attr(docsrs, doc(cfg(feature = "onnx")))]
pub use ort::session::builder::GraphOptimizationLevel;

use crate::error::{Error, Result};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[cfg(all(feature = "serde", feature = "onnx"))]
mod graph_optimization_level {
  use super::GraphOptimizationLevel;
  use serde::*;

  #[derive(
    Debug, Default, Clone, Copy, Eq, PartialEq, Hash, Ord, PartialOrd, Serialize, Deserialize,
  )]
  #[serde(rename_all = "snake_case")]
  enum OptimizationLevel {
    Disable,
    Level1,
    Level2,
    #[default]
    Level3,
    All,
  }

  impl From<GraphOptimizationLevel> for OptimizationLevel {
    #[inline]
    fn from(value: GraphOptimizationLevel) -> Self {
      match value {
        GraphOptimizationLevel::Disable => Self::Disable,
        GraphOptimizationLevel::Level1 => Self::Level1,
        GraphOptimizationLevel::Level2 => Self::Level2,
        GraphOptimizationLevel::Level3 => Self::Level3,
        GraphOptimizationLevel::All => Self::All,
      }
    }
  }

  impl From<OptimizationLevel> for GraphOptimizationLevel {
    #[inline]
    fn from(value: OptimizationLevel) -> Self {
      match value {
        OptimizationLevel::Disable => Self::Disable,
        OptimizationLevel::Level1 => Self::Level1,
        OptimizationLevel::Level2 => Self::Level2,
        OptimizationLevel::Level3 => Self::Level3,
        OptimizationLevel::All => Self::All,
      }
    }
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn serialize<S>(level: &GraphOptimizationLevel, serializer: S) -> Result<S::Ok, S::Error>
  where
    S: Serializer,
  {
    OptimizationLevel::from(*level).serialize(serializer)
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn deserialize<'de, D>(deserializer: D) -> Result<GraphOptimizationLevel, D::Error>
  where
    D: Deserializer<'de>,
  {
    OptimizationLevel::deserialize(deserializer).map(Into::into)
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn default() -> GraphOptimizationLevel {
    GraphOptimizationLevel::Disable
  }
}

/// Sample rates directly supported by the Silero VAD model.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
pub enum SampleRate {
  /// 8 kHz sample rate, which uses smaller chunks and less context.
  #[cfg_attr(feature = "serde", serde(rename = "8k"))]
  Rate8k,
  /// 16 kHz sample rate, which uses larger chunks and more context for better accuracy.
  #[cfg_attr(feature = "serde", serde(rename = "16k"))]
  #[default]
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
/// Options for constructing an ONNX session.
///
/// This type intentionally stays small. Deployment-specific runtime
/// policy such as `intra_threads` / `inter_threads` should normally be
/// configured one layer up, then passed down via
/// [`crate::Session::from_ort_session`].
#[cfg(feature = "onnx")]
#[cfg_attr(docsrs, doc(cfg(feature = "onnx")))]
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SessionOptions {
  #[cfg_attr(
    feature = "serde",
    serde(
      default = "graph_optimization_level::default",
      with = "graph_optimization_level"
    )
  )]
  optimization_level: GraphOptimizationLevel,
}

#[cfg(feature = "onnx")]
impl Default for SessionOptions {
  #[inline]
  fn default() -> Self {
    Self::new()
  }
}

#[cfg(feature = "onnx")]
impl SessionOptions {
  /// Create a new `SessionOptions` with default values.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn new() -> Self {
    Self {
      optimization_level: GraphOptimizationLevel::Level3,
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

#[cfg_attr(not(tarpaulin), inline(always))]
const fn default_start_threshold() -> f32 {
  0.5
}

#[cfg_attr(not(tarpaulin), inline(always))]
const fn default_min_speech_duration() -> Duration {
  Duration::from_millis(250)
}

#[cfg_attr(not(tarpaulin), inline(always))]
const fn default_min_silence_duration() -> Duration {
  Duration::from_millis(100)
}

#[cfg_attr(not(tarpaulin), inline(always))]
const fn default_min_silence_at_max_speech() -> Duration {
  Duration::from_millis(98)
}

#[cfg_attr(not(tarpaulin), inline(always))]
const fn default_speech_pad() -> Duration {
  Duration::from_millis(30)
}

/// Configuration for turning frame probabilities into speech segments.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SpeechOptions {
  #[cfg_attr(feature = "serde", serde(default))]
  sample_rate: SampleRate,
  #[cfg_attr(feature = "serde", serde(default = "default_start_threshold"))]
  start_threshold: f32,
  #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
  end_threshold: Option<f32>,
  #[cfg_attr(
    feature = "serde",
    serde(default = "default_min_speech_duration", with = "humantime_serde")
  )]
  min_speech_duration: Duration,
  #[cfg_attr(
    feature = "serde",
    serde(default = "default_min_silence_duration", with = "humantime_serde")
  )]
  min_silence_duration: Duration,
  #[cfg_attr(
    feature = "serde",
    serde(
      default = "default_min_silence_at_max_speech",
      with = "humantime_serde"
    )
  )]
  min_silence_at_max_speech: Duration,
  #[cfg_attr(
    feature = "serde",
    serde(
      skip_serializing_if = "Option::is_none",
      with = "humantime_serde::option"
    )
  )]
  max_speech_duration: Option<Duration>,
  #[cfg_attr(
    feature = "serde",
    serde(default = "default_speech_pad", with = "humantime_serde")
  )]
  speech_pad: Duration,
}

impl Default for SpeechOptions {
  #[cfg_attr(not(tarpaulin), inline(always))]
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
      start_threshold: default_start_threshold(),
      end_threshold: None,
      min_speech_duration: default_min_speech_duration(),
      min_silence_duration: default_min_silence_duration(),
      // Matches the upstream silero-vad Python default (0.098 s).
      min_silence_at_max_speech: default_min_silence_at_max_speech(),
      max_speech_duration: None,
      speech_pad: default_speech_pad(),
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

  /// Returns the effective end threshold.
  ///
  /// If a user-supplied end threshold would break the hysteresis
  /// window, this falls back to the same derived threshold used by the
  /// default configuration so behavior stays stable regardless of
  /// builder call order.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn end_threshold(&self) -> f32 {
    effective_end_threshold(
      self.start_threshold,
      self
        .end_threshold
        .unwrap_or_else(|| default_end_threshold(self.start_threshold)),
    )
  }

  /// Returns the minimum duration of detected speech segments, in milliseconds.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn min_speech_duration(&self) -> Duration {
    self.min_speech_duration
  }

  /// Returns the minimum duration of silence required to close a detected speech segment, in milliseconds.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn min_silence_duration(&self) -> Duration {
    self.min_silence_duration
  }

  /// Returns the minimum silence duration used as a preferred split point when the maximum speech duration is reached.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn min_silence_at_max_speech(&self) -> Duration {
    self.min_silence_at_max_speech
  }

  /// Returns the maximum duration of a speech segment before the segmenter force-splits it.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn max_speech_duration(&self) -> Option<Duration> {
    self.max_speech_duration
  }

  /// Returns the amount of padding to add to the start of detected speech segments, in milliseconds.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn speech_pad(&self) -> Duration {
    self.speech_pad
  }

  /// Returns the minimum duration of detected speech segments, in samples.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn min_speech_samples(&self) -> u64 {
    ms_to_samples(self.min_speech_duration, self.sample_rate)
  }

  /// Returns the minimum duration of silence required to close a detected speech segment, in samples.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn min_silence_samples(&self) -> u64 {
    ms_to_samples(self.min_silence_duration, self.sample_rate)
  }

  /// Returns the minimum silence duration used as a preferred split point when max speech duration is reached, in samples.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn min_silence_at_max_speech_samples(&self) -> u64 {
    ms_to_samples(self.min_silence_at_max_speech, self.sample_rate)
  }

  /// Returns the maximum speech duration before force-splitting, in samples.
  ///
  /// This matches the upstream silero-vad derivation:
  /// - `- chunk_samples` because the split check runs on the next frame after
  ///   the limit is exceeded
  /// - `- 2 * speech_pad_samples` because emitted segments pad both the end of
  ///   the current segment and the start of the next one
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn max_speech_samples(&self) -> Option<u64> {
    self.max_speech_duration.map(|duration| {
      ms_to_samples(duration, self.sample_rate)
        .saturating_sub(self.sample_rate.chunk_samples() as u64)
        .saturating_sub(self.speech_pad_samples().saturating_mul(2))
    })
  }

  /// Returns the amount of padding to add to the start of detected speech segments, in samples.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn speech_pad_samples(&self) -> u64 {
    ms_to_samples(self.speech_pad, self.sample_rate)
  }

  /// Set the sample rate to use for speech detection.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_sample_rate(mut self, sample_rate: SampleRate) -> Self {
    self.set_sample_rate(sample_rate);
    self
  }

  /// Set the start threshold, which must be between 0 and 1. If not set, it defaults to 0.5.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_start_threshold(mut self, threshold: f32) -> Self {
    self.set_start_threshold(threshold);
    self
  }

  /// Set the preferred end threshold.
  ///
  /// The stored value is sanitized into the `[0, 1]` range. When the
  /// threshold is later read via [`Self::end_threshold`], it is also
  /// checked against the current start threshold. Invalid combinations
  /// fall back to the default derived hysteresis rule even if builder
  /// methods are called in a different order.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_end_threshold(mut self, threshold: f32) -> Self {
    self.set_end_threshold(threshold);
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
  pub const fn with_min_speech_duration(mut self, duration: Duration) -> Self {
    self.set_min_speech_duration(duration);
    self
  }

  /// Set the minimum duration of silence required to close a detected speech segment, in milliseconds.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_min_silence_duration(mut self, duration: Duration) -> Self {
    self.set_min_silence_duration(duration);
    self
  }

  /// Set the minimum silence duration that can be used as a preferred split point when maximum speech duration is reached.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_min_silence_at_max_speech(mut self, duration: Duration) -> Self {
    self.set_min_silence_at_max_speech(duration);
    self
  }

  /// Set the maximum duration of a speech segment before the segmenter force-splits it.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_max_speech_duration(mut self, duration: Duration) -> Self {
    self.set_max_speech_duration(duration);
    self
  }

  /// Clear the maximum speech duration, disabling force-splitting by segment length.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn clear_max_speech_duration(mut self) -> Self {
    self.max_speech_duration = None;
    self
  }

  /// Set the amount of padding to add to the start of detected speech segments, in milliseconds.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_speech_pad(mut self, pad: Duration) -> Self {
    self.set_speech_pad(pad);
    self
  }

  /// Set the sample rate to use for speech detection.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_sample_rate(&mut self, sample_rate: SampleRate) -> &mut Self {
    self.sample_rate = sample_rate;
    self
  }

  /// Set the start threshold, which must be between 0 and 1. If not set, it defaults to 0.5.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_start_threshold(&mut self, threshold: f32) -> &mut Self {
    self.start_threshold = sanitize_probability(threshold);
    self
  }

  /// Set the preferred end threshold.
  ///
  /// The stored value is sanitized into the `[0, 1]` range. When the
  /// threshold is later read via [`Self::end_threshold`], it is also
  /// checked against the current start threshold. Invalid combinations
  /// fall back to the default derived hysteresis rule even if builder
  /// methods are called in a different order.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_end_threshold(&mut self, threshold: f32) -> &mut Self {
    self.end_threshold = Some(sanitize_probability(threshold));
    self
  }

  /// Set the minimum duration of detected speech segments as a `Duration`.
  /// Sub-second precision is supported according to the precision of `Duration`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_min_speech_duration(&mut self, duration: Duration) -> &mut Self {
    self.min_speech_duration = duration;
    self
  }

  /// Set the minimum duration of silence required to close a detected speech segment as a `Duration`.
  /// Sub-second precision is supported according to the precision of `Duration`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_min_silence_duration(&mut self, duration: Duration) -> &mut Self {
    self.min_silence_duration = duration;
    self
  }

  /// Set the minimum silence duration, as a `Duration`, that can be used as a preferred split point
  /// when maximum speech duration is reached. Sub-second precision is supported according to the
  /// precision of `Duration`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_min_silence_at_max_speech(&mut self, duration: Duration) -> &mut Self {
    self.min_silence_at_max_speech = duration;
    self
  }

  /// Set the maximum duration of a speech segment, as a `Duration`, before the segmenter
  /// force-splits it. Sub-second precision is supported according to the precision of `Duration`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_max_speech_duration(&mut self, duration: Duration) -> &mut Self {
    self.max_speech_duration = Some(duration);
    self
  }

  /// Set the amount of padding to add to the start of detected speech segments, in milliseconds.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_speech_pad(&mut self, pad: Duration) -> &mut Self {
    self.speech_pad = pad;
    self
  }
}

#[inline]
pub(crate) const fn ms_to_samples(duration: Duration, sample_rate: SampleRate) -> u64 {
  let samples = (duration.as_millis() * (sample_rate.hz() as u128)) / 1_000;

  if samples > u64::MAX as u128 {
    u64::MAX
  } else {
    samples as u64
  }
}

#[inline]
const fn sanitize_probability(value: f32) -> f32 {
  if value.is_finite() {
    value.clamp(0.0, 1.0)
  } else {
    0.0
  }
}

#[inline]
const fn default_end_threshold(start_threshold: f32) -> f32 {
  sanitize_probability((sanitize_probability(start_threshold) - 0.15).max(0.01))
}

#[inline]
const fn effective_end_threshold(start_threshold: f32, end_threshold: f32) -> f32 {
  let start_threshold = sanitize_probability(start_threshold);
  let end_threshold = sanitize_probability(end_threshold);

  if end_threshold < start_threshold {
    end_threshold
  } else {
    default_end_threshold(start_threshold)
  }
}

#[cfg(test)]
mod tests {
  use std::time::Duration;

  #[cfg(feature = "onnx")]
  use ort::session::builder::GraphOptimizationLevel;

  #[cfg(feature = "onnx")]
  use super::SessionOptions;
  use super::{SampleRate, SpeechOptions, ms_to_samples};

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
    assert_eq!(config.min_speech_duration(), Duration::from_millis(250));
    assert_eq!(config.min_silence_duration(), Duration::from_millis(100));
    assert_eq!(
      config.min_silence_at_max_speech(),
      Duration::from_millis(98)
    );
    assert_eq!(config.max_speech_duration(), None);
    assert_eq!(config.speech_pad(), Duration::from_millis(30));
  }

  #[test]
  fn ms_to_samples_uses_stream_rate() {
    assert_eq!(
      ms_to_samples(Duration::from_millis(100), SampleRate::Rate16k),
      1_600
    );
    assert_eq!(
      ms_to_samples(Duration::from_millis(100), SampleRate::Rate8k),
      800
    );
  }

  #[cfg(feature = "onnx")]
  #[test]
  fn session_options_default_to_unopinionated_core_settings() {
    let options = SessionOptions::default();
    assert_eq!(options.optimization_level(), GraphOptimizationLevel::Level3,);
  }

  #[test]
  fn end_threshold_falls_back_to_default_gap_when_builder_order_would_invert_hysteresis() {
    let options = SpeechOptions::default()
      .with_start_threshold(0.4)
      .with_end_threshold(0.6);
    assert!(options.end_threshold() < options.start_threshold());
    assert!((options.end_threshold() - 0.25).abs() < f32::EPSILON);

    let reordered = SpeechOptions::default()
      .with_end_threshold(0.6)
      .with_start_threshold(0.4);
    assert!(reordered.end_threshold() < reordered.start_threshold());
    assert!((options.end_threshold() - reordered.end_threshold()).abs() < f32::EPSILON);

    let valid = SpeechOptions::default()
      .with_start_threshold(0.6)
      .with_end_threshold(0.2);
    assert!((valid.end_threshold() - 0.2).abs() < f32::EPSILON);
  }

  #[test]
  fn max_speech_duration_converts_to_samples_with_stream_lookahead_and_padding() {
    let options = SpeechOptions::default()
      .with_speech_pad(Duration::from_millis(30))
      .with_max_speech_duration(Duration::from_millis(1_000));
    assert_eq!(
      options.max_speech_duration(),
      Some(Duration::from_millis(1_000))
    );
    assert_eq!(options.min_silence_at_max_speech_samples(), 1_568);
    assert_eq!(options.max_speech_samples(), Some(14_528));
  }

  #[cfg(all(feature = "serde", feature = "onnx"))]
  #[test]
  fn test_serde() {
    let opts = SessionOptions::default().with_optimization_level(GraphOptimizationLevel::Level2);
    let serialized = serde_json::to_string(&opts).expect("serialize options");
    let deserialized: SessionOptions =
      serde_json::from_str(&serialized).expect("deserialize options");
    assert_eq!(opts.optimization_level, deserialized.optimization_level);

    let default_deserialized: SessionOptions =
      serde_json::from_str("{}").expect("deserialize default options");
    assert!(matches!(
      default_deserialized.optimization_level,
      GraphOptimizationLevel::Disable
    ));

    // level1
    let level1_opts =
      SessionOptions::default().with_optimization_level(GraphOptimizationLevel::Level1);
    let level1_serialized = serde_json::to_string(&level1_opts).expect("serialize level1 options");
    let level1_deserialized: SessionOptions =
      serde_json::from_str(&level1_serialized).expect("deserialize level1 options");
    assert!(matches!(
      level1_deserialized.optimization_level,
      GraphOptimizationLevel::Level1
    ));

    // level2
    let level2_opts =
      SessionOptions::default().with_optimization_level(GraphOptimizationLevel::Level2);
    let level2_serialized = serde_json::to_string(&level2_opts).expect("serialize level2 options");
    let level2_deserialized: SessionOptions =
      serde_json::from_str(&level2_serialized).expect("deserialize level2 options");
    assert!(matches!(
      level2_deserialized.optimization_level,
      GraphOptimizationLevel::Level2
    ));

    // level3
    let level3_opts =
      SessionOptions::default().with_optimization_level(GraphOptimizationLevel::Level3);
    let level3_serialized = serde_json::to_string(&level3_opts).expect("serialize level3 options");
    let level3_deserialized: SessionOptions =
      serde_json::from_str(&level3_serialized).expect("deserialize level3 options");
    assert!(matches!(
      level3_deserialized.optimization_level,
      GraphOptimizationLevel::Level3
    ));

    // all
    let all_opts = SessionOptions::default().with_optimization_level(GraphOptimizationLevel::All);
    let all_serialized = serde_json::to_string(&all_opts).expect("serialize all options");
    let all_deserialized: SessionOptions =
      serde_json::from_str(&all_serialized).expect("deserialize all options");
    assert!(matches!(
      all_deserialized.optimization_level,
      GraphOptimizationLevel::All
    ));

    // disable
    let disable_opts =
      SessionOptions::default().with_optimization_level(GraphOptimizationLevel::Disable);
    let disable_serialized =
      serde_json::to_string(&disable_opts).expect("serialize disable options");
    let disable_deserialized: SessionOptions =
      serde_json::from_str(&disable_serialized).expect("deserialize disable options");
    assert!(matches!(
      disable_deserialized.optimization_level,
      GraphOptimizationLevel::Disable
    ));
  }
}
