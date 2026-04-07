use crate::{
  Result, Session, StreamState,
  error::Error,
  options::{SampleRate, SpeechOptions},
};

/// One speech segment on the stream timeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpeechSegment {
  start_sample: u64,
  end_sample: u64,
  sample_rate: SampleRate,
}

impl SpeechSegment {
  /// Create a new speech segment with the given start and end samples and sample rate.
  #[inline]
  pub const fn new(start_sample: u64, end_sample: u64, sample_rate: SampleRate) -> Self {
    Self {
      start_sample,
      end_sample,
      sample_rate,
    }
  }

  /// Returns the start sample of this speech segment.
  #[inline]
  pub const fn start_sample(&self) -> u64 {
    self.start_sample
  }

  /// Returns the end sample of this speech segment.
  #[inline]
  pub const fn end_sample(&self) -> u64 {
    self.end_sample
  }

  /// Returns the sample rate of this speech segment.
  #[inline]
  pub const fn sample_rate(&self) -> SampleRate {
    self.sample_rate
  }

  /// Returns the number of samples in this speech segment.
  #[inline]
  pub const fn sample_count(&self) -> u64 {
    self.end_sample.saturating_sub(self.start_sample)
  }

  /// Returns the start time of this speech segment in seconds.
  #[inline]
  pub fn start_seconds(&self) -> f64 {
    self.start_sample as f64 / self.sample_rate.hz() as f64
  }

  /// Returns the end time of this speech segment in seconds.
  #[inline]
  pub fn end_seconds(&self) -> f64 {
    self.end_sample as f64 / self.sample_rate.hz() as f64
  }
}

/// Streaming post-processor that turns frame probabilities into
/// speech segments.
///
/// The segmenter is intentionally model-agnostic: it only consumes
/// frame probabilities. This lets higher-level runtimes choose between
/// single-stream inference and micro-batched inference while still
/// reusing the same segment semantics.
#[derive(Debug, Clone)]
pub struct SpeechSegmenter {
  options: SpeechOptions,
  current_sample: u64,
  active_start: Option<u64>,
  tentative_end: Option<u64>,
  // Start sample of the most recent silence long enough to be a preferred
  // force-split point.
  max_split_end: Option<u64>,
  // First speech frame after `max_split_end`; used to resume after a
  // force-split at that silence boundary.
  next_start: Option<u64>,
}

impl SpeechSegmenter {
  /// Create a new `SpeechSegmenter` with the given options.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn new(options: SpeechOptions) -> Self {
    Self {
      options,
      current_sample: 0,
      active_start: None,
      tentative_end: None,
      max_split_end: None,
      next_start: None,
    }
  }

  /// Returns a reference to the `SpeechOptions` used by this segmenter.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn options(&self) -> &SpeechOptions {
    &self.options
  }

  /// Reconfigure the segmenter for a stream with a different sample rate.
  ///
  /// Changing sample rate starts a new logical timeline, so any
  /// in-flight segment state is cleared.
  #[inline]
  pub fn set_sample_rate(&mut self, sample_rate: SampleRate) {
    if self.sample_rate() != sample_rate {
      self.options = self.options.with_sample_rate(sample_rate);
      self.reset();
    }
  }

  /// Returns the sample rate used by this segmenter.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn sample_rate(&self) -> SampleRate {
    self.options.sample_rate()
  }

  /// Returns whether the segmenter is currently active (i.e., has an ongoing speech segment).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn is_active(&self) -> bool {
    self.active_start.is_some()
  }

  /// Reset the segmenter state, clearing any ongoing segments and pending samples.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn reset(&mut self) {
    self.current_sample = 0;
    self.active_start = None;
    self.tentative_end = None;
    self.max_split_end = None;
    self.next_start = None;
  }

  /// Consume one probability for one Silero frame.
  ///
  /// Returns `Some(segment)` only when a speech segment can be closed
  /// with the currently available evidence.
  pub fn push_probability(&mut self, probability: f32) -> Option<SpeechSegment> {
    let frame_samples = self.sample_rate().chunk_samples() as u64;
    let frame_start = self.current_sample;
    self.current_sample = self.current_sample.saturating_add(frame_samples);

    if probability >= self.options.start_threshold() {
      if let Some(tentative_end) = self.tentative_end.take() {
        let silence_samples = frame_start.saturating_sub(tentative_end);
        if silence_samples > self.options.min_silence_at_max_speech_samples() {
          self.max_split_end = Some(tentative_end);
          self.next_start = Some(frame_start);
        }
      }
      if self.active_start.is_none() {
        self.active_start = Some(frame_start.saturating_sub(self.options.speech_pad_samples()));
        return None;
      }
    }

    let start = self.active_start?;
    if let Some(max_speech_samples) = self.options.max_speech_samples() {
      if frame_start.saturating_sub(start) > max_speech_samples {
        return self.split_at_max_duration(frame_start, probability);
      }
    }

    if probability >= self.options.end_threshold() {
      self.tentative_end = None;
      return None;
    }

    let silence_start = *self.tentative_end.get_or_insert(frame_start);
    let silence_samples = self.current_sample.saturating_sub(silence_start);
    if silence_samples > self.options.min_silence_at_max_speech_samples() {
      self.max_split_end = Some(silence_start);
    }
    if silence_samples < self.options.min_silence_samples() {
      return None;
    }

    self.clear_segment_memory();
    self.build_segment(start, silence_start)
  }

  /// Process a buffer of audio samples, emitting speech segments as they are detected.
  pub fn process_samples<F>(
    &mut self,
    session: &mut Session,
    stream: &mut StreamState,
    samples: &[f32],
    mut emit: F,
  ) -> Result<usize>
  where
    F: FnMut(SpeechSegment),
  {
    self.ensure_sample_rate(stream.sample_rate())?;
    session.process_stream(stream, samples, |probability| {
      if let Some(segment) = self.push_probability(probability) {
        emit(segment);
      }
    })
  }

  /// Flush any remaining pending samples for a stream, emitting a final speech segment if the flushed tail confirms the end of an active segment.
  pub fn flush_stream<F>(
    &mut self,
    session: &mut Session,
    stream: &mut StreamState,
    mut emit: F,
  ) -> Result<()>
  where
    F: FnMut(SpeechSegment),
  {
    self.ensure_sample_rate(stream.sample_rate())?;
    if let Some(probability) = session.flush_stream(stream)? {
      if let Some(segment) = self.push_probability(probability) {
        emit(segment);
      }
    }
    Ok(())
  }

  /// Finish the current stream and emit any trailing open segment.
  ///
  /// This resets the segmenter so it can be reused for a new stream.
  pub fn finish(&mut self) -> Option<SpeechSegment> {
    let trailing = self.active_start.and_then(|start| {
      let end = self.current_sample;
      if end.saturating_sub(start) < self.options.min_speech_samples() {
        None
      } else {
        Some(SpeechSegment::new(start, end, self.sample_rate()))
      }
    });
    self.reset();
    trailing
  }

  /// Convenience for end-of-stream handling: flush the model tail and
  /// then close any trailing open segment.
  pub fn finish_stream<F>(
    &mut self,
    session: &mut Session,
    stream: &mut StreamState,
    mut emit: F,
  ) -> Result<()>
  where
    F: FnMut(SpeechSegment),
  {
    self.flush_stream(session, stream, &mut emit)?;
    if let Some(segment) = self.finish() {
      emit(segment);
    }
    Ok(())
  }

  fn ensure_sample_rate(&self, sample_rate: SampleRate) -> Result<()> {
    if self.sample_rate() == sample_rate {
      Ok(())
    } else {
      Err(Error::IncompatibleSampleRate {
        expected: self.sample_rate().hz(),
        actual: sample_rate.hz(),
      })
    }
  }

  fn split_at_max_duration(&mut self, frame_start: u64, probability: f32) -> Option<SpeechSegment> {
    let start = self.active_start?;
    let raw_end = self.max_split_end.unwrap_or(frame_start);
    let segment = self.build_segment(start, raw_end);

    self.active_start = if let Some(next_start) = self.next_start.filter(|next| *next >= raw_end) {
      Some(next_start.saturating_sub(self.options.speech_pad_samples()))
    } else if self.max_split_end.is_none() && probability >= self.options.start_threshold() {
      Some(frame_start.saturating_sub(self.options.speech_pad_samples()))
    } else {
      None
    };
    self.clear_split_tracking();

    segment
  }

  fn build_segment(&self, start: u64, raw_end: u64) -> Option<SpeechSegment> {
    let end_sample = raw_end
      .saturating_add(self.options.speech_pad_samples())
      .min(self.current_sample);
    if end_sample.saturating_sub(start) < self.options.min_speech_samples() {
      None
    } else {
      Some(SpeechSegment::new(start, end_sample, self.sample_rate()))
    }
  }

  fn clear_segment_memory(&mut self) {
    self.active_start = None;
    self.clear_split_tracking();
  }

  fn clear_split_tracking(&mut self) {
    self.tentative_end = None;
    self.max_split_end = None;
    self.next_start = None;
  }
}

/// Backwards-compatible alias for callers that think in
/// "detector" rather than "segmenter" terms.
pub type SpeechDetector = SpeechSegmenter;

/// Convenience helper for one-shot offline detection on a full buffer.
pub fn detect_speech(
  session: &mut Session,
  samples: &[f32],
  config: SpeechOptions,
) -> Result<Vec<SpeechSegment>> {
  let mut stream = StreamState::new(config.sample_rate());
  let mut segmenter = SpeechSegmenter::new(config);
  let mut segments = Vec::new();
  segmenter.process_samples(session, &mut stream, samples, |segment| {
    segments.push(segment)
  })?;
  segmenter.finish_stream(session, &mut stream, |segment| segments.push(segment))?;
  Ok(segments)
}

#[cfg(test)]
mod tests {
  use crate::{SampleRate, SpeechOptions};

  use super::{SpeechSegment, SpeechSegmenter};

  fn frame_count(duration_ms: u32, sample_rate: SampleRate) -> usize {
    let frame_ms = (sample_rate.chunk_samples() as u32 * 1_000) / sample_rate.hz();
    (duration_ms / frame_ms) as usize
  }

  fn collect(segmenter: &mut SpeechSegmenter, probabilities: &[f32]) -> Vec<SpeechSegment> {
    let mut segments = Vec::new();
    for probability in probabilities {
      if let Some(segment) = segmenter.push_probability(*probability) {
        segments.push(segment);
      }
    }
    if let Some(segment) = segmenter.finish() {
      segments.push(segment);
    }
    segments
  }

  #[test]
  fn closes_segment_after_confirmed_silence() {
    let config = SpeechOptions::default();
    let mut segmenter = SpeechSegmenter::new(config);
    let mut probabilities = vec![0.9; frame_count(320, SampleRate::Rate16k)];
    probabilities.extend(vec![0.0; frame_count(128, SampleRate::Rate16k)]);

    let segments = collect(&mut segmenter, &probabilities);
    assert_eq!(segments.len(), 1);
    assert!(segments[0].start_sample() <= config.speech_pad_samples());
    assert!(segments[0].sample_count() >= config.min_speech_samples());
  }

  #[test]
  fn drops_short_bursts() {
    let config = SpeechOptions::default();
    let mut segmenter = SpeechSegmenter::new(config);
    let mut probabilities = vec![0.9; frame_count(64, SampleRate::Rate16k)];
    probabilities.extend(vec![0.0; frame_count(160, SampleRate::Rate16k)]);
    let segments = collect(&mut segmenter, &probabilities);
    assert!(segments.is_empty());
  }

  #[test]
  fn finish_flushes_trailing_active_segment() {
    let config = SpeechOptions::default();
    let mut segmenter = SpeechSegmenter::new(config);
    let probabilities = vec![0.9; frame_count(320, SampleRate::Rate16k)];
    let segments = collect(&mut segmenter, &probabilities);
    assert_eq!(segments.len(), 1);
    assert!(segments[0].end_sample() > segments[0].start_sample());
  }

  #[test]
  fn reset_clears_runtime_state() {
    let mut segmenter = SpeechSegmenter::new(SpeechOptions::default());
    let _ = segmenter.push_probability(0.9);
    assert!(segmenter.is_active());
    segmenter.reset();
    assert!(!segmenter.is_active());
  }

  #[test]
  fn set_sample_rate_resets_runtime_state_and_updates_timeline_rate() {
    let mut segmenter = SpeechSegmenter::new(SpeechOptions::default());
    let _ = segmenter.push_probability(0.9);
    assert!(segmenter.is_active());

    segmenter.set_sample_rate(SampleRate::Rate8k);
    assert_eq!(segmenter.sample_rate(), SampleRate::Rate8k);
    assert!(!segmenter.is_active());

    for _ in 0..frame_count(320, SampleRate::Rate8k) {
      let _ = segmenter.push_probability(0.9);
    }
    let segment = segmenter.finish().expect("trailing segment");
    assert_eq!(segment.sample_rate(), SampleRate::Rate8k);
  }

  #[test]
  fn force_splits_long_speech_when_max_duration_is_reached() {
    let config = SpeechOptions::default()
      .with_min_speech_duration_ms(0)
      .with_speech_pad_ms(0)
      .with_max_speech_duration_ms(160);
    let mut segmenter = SpeechSegmenter::new(config);
    let probabilities = vec![0.9; 8];

    let segments = collect(&mut segmenter, &probabilities);
    assert_eq!(segments.len(), 2);
    assert_eq!(segments[0].start_sample(), 0);
    assert_eq!(segments[0].end_sample(), 2_560);
    assert_eq!(segments[1].start_sample(), 2_560);
    assert_eq!(segments[1].end_sample(), 4_096);
  }

  #[test]
  fn prefers_recorded_silence_when_splitting_long_speech() {
    let config = SpeechOptions::default()
      .with_min_speech_duration_ms(0)
      .with_speech_pad_ms(0)
      .with_min_silence_duration_ms(300)
      .with_min_silence_at_max_speech_ms(64)
      .with_max_speech_duration_ms(256);
    let mut segmenter = SpeechSegmenter::new(config);
    let mut probabilities = vec![0.9; 4];
    probabilities.extend(vec![0.0; 4]);
    probabilities.extend(vec![0.9; 4]);

    let segments = collect(&mut segmenter, &probabilities);
    assert_eq!(segments.len(), 2);
    assert_eq!(segments[0].start_sample(), 0);
    assert_eq!(segments[0].end_sample(), 2_048);
    assert_eq!(segments[1].start_sample(), 4_096);
    assert_eq!(segments[1].end_sample(), 6_144);
  }

  #[test]
  fn non_qualifying_silence_does_not_overwrite_next_start() {
    let config = SpeechOptions::default()
      .with_min_speech_duration_ms(0)
      .with_speech_pad_ms(0)
      .with_min_silence_duration_ms(10_000)
      .with_min_silence_at_max_speech_ms(64)
      .with_max_speech_duration_ms(512);
    let mut segmenter = SpeechSegmenter::new(config);

    let mut probabilities = vec![0.9; 4];
    probabilities.extend(vec![0.0; 4]);
    probabilities.extend(vec![0.9; 4]);
    probabilities.extend(vec![0.0; 1]);
    probabilities.extend(vec![0.9; 20]);

    let segments = collect(&mut segmenter, &probabilities);
    assert_eq!(segments[0].end_sample(), 2_048);
    assert_eq!(segments[1].start_sample(), 4_096);
  }

  #[test]
  fn force_split_during_silence_closes_without_restarting() {
    let config = SpeechOptions::default()
      .with_min_speech_duration_ms(0)
      .with_speech_pad_ms(0)
      .with_min_silence_duration_ms(10_000)
      .with_min_silence_at_max_speech_ms(64)
      .with_max_speech_duration_ms(224);
    let mut segmenter = SpeechSegmenter::new(config);

    let mut probabilities = vec![0.9; 4];
    probabilities.extend(vec![0.0; 8]);

    let segments = collect(&mut segmenter, &probabilities);
    assert_eq!(segments.len(), 1);
    assert_eq!(segments[0].start_sample(), 0);
    assert_eq!(segments[0].end_sample(), 2_048);
  }

  #[test]
  fn force_split_applies_speech_pad_to_split_boundaries() {
    let config = SpeechOptions::default()
      .with_min_speech_duration_ms(0)
      .with_speech_pad_ms(32)
      .with_min_silence_duration_ms(10_000)
      .with_min_silence_at_max_speech_ms(64)
      .with_max_speech_duration_ms(512);
    let mut segmenter = SpeechSegmenter::new(config);

    let mut probabilities = vec![0.9; 4];
    probabilities.extend(vec![0.0; 4]);
    probabilities.extend(vec![0.9; 8]);

    let segments = collect(&mut segmenter, &probabilities);
    assert_eq!(segments[0].end_sample(), 2_560);
    assert_eq!(segments[1].start_sample(), 3_584);
  }
}
