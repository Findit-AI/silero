use std::collections::VecDeque;

use crate::{
  Result,
  backend::VadBackend,
  options::{SampleRate, SpeechOptions},
};
#[cfg(feature = "onnx")]
use crate::{Session, StreamState, error::Error};

/// One speech segment on the stream timeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpeechSegment {
  start_sample: u64,
  end_sample: u64,
  sample_rate: SampleRate,
}

impl SpeechSegment {
  /// Create a new speech segment with the given start and end samples and sample rate.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn new(start_sample: u64, end_sample: u64, sample_rate: SampleRate) -> Self {
    Self {
      start_sample,
      end_sample,
      sample_rate,
    }
  }

  /// Returns the start sample of this speech segment.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn start_sample(&self) -> u64 {
    self.start_sample
  }

  /// Returns the end sample of this speech segment.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn end_sample(&self) -> u64 {
    self.end_sample
  }

  /// Returns the sample rate of this speech segment.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn sample_rate(&self) -> SampleRate {
    self.sample_rate
  }

  /// Returns the number of samples in this speech segment.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn sample_count(&self) -> u64 {
    self.end_sample.saturating_sub(self.start_sample)
  }

  /// Returns the start time of this speech segment in seconds.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn start_seconds(&self) -> f64 {
    self.start_sample as f64 / self.sample_rate.hz() as f64
  }

  /// Returns the end time of this speech segment in seconds.
  #[cfg_attr(not(tarpaulin), inline(always))]
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
///
/// Its timeline advances by [`frame_samples`](Self::frame_samples) per
/// probability. That defaults to the sample rate's model chunk size
/// (512 samples at 16 kHz — the ONNX geometry), but a backend that
/// declares a different frame size drives it through
/// [`set_frame_samples`](Self::set_frame_samples) (done automatically by
/// [`detect_speech_with`]) so the same segmentation rules apply at any
/// frame geometry.
#[derive(Debug, Clone)]
pub struct SpeechSegmenter {
  options: SpeechOptions,
  // Samples per model frame; the timeline advances by this per
  // probability. Defaults to `sample_rate().chunk_samples()`; a
  // non-ONNX backend overrides it via `set_frame_samples`.
  frame_samples: u64,
  current_sample: u64,
  // Padded start sample used for emitted segments.
  active_start: Option<u64>,
  // Raw model-frame start sample used for upstream-compatible duration checks.
  active_raw_start: Option<u64>,
  tentative_end: Option<u64>,
  // Start sample of the most recent silence long enough to be a preferred
  // force-split point.
  max_split_end: Option<u64>,
  // First speech frame after `max_split_end`; used to resume after a
  // force-split at that silence boundary.
  next_start: Option<u64>,
  // Queue of segments closed by recent push_samples / finish_stream calls
  // that have not yet been popped by the caller. Drained one segment at
  // a time via `push_samples(&[])`.
  pending_segments: VecDeque<SpeechSegment>,
}

impl SpeechSegmenter {
  /// Create a new `SpeechSegmenter` with the given options.
  ///
  /// The frame geometry defaults to the options' sample-rate model chunk
  /// size (512 samples at 16 kHz). Override it with
  /// [`set_frame_samples`](Self::set_frame_samples) for a backend that
  /// declares a different frame size.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn new(options: SpeechOptions) -> Self {
    let frame_samples = options.sample_rate().chunk_samples() as u64;
    Self {
      options,
      frame_samples,
      current_sample: 0,
      active_start: None,
      active_raw_start: None,
      tentative_end: None,
      max_split_end: None,
      next_start: None,
      pending_segments: VecDeque::new(),
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
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_sample_rate(&mut self, sample_rate: SampleRate) {
    if self.sample_rate() != sample_rate {
      self.options.set_sample_rate(sample_rate);
      // A new rate implies a new default frame geometry; a custom
      // backend frame size must be re-applied via `set_frame_samples`.
      self.frame_samples = sample_rate.chunk_samples() as u64;
      self.reset();
    }
  }

  /// Returns the sample rate used by this segmenter.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn sample_rate(&self) -> SampleRate {
    self.options.sample_rate()
  }

  /// Returns the number of samples per model frame that this segmenter's
  /// timeline advances by per probability.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn frame_samples(&self) -> usize {
    self.frame_samples as usize
  }

  /// Set the number of samples per model frame.
  ///
  /// The segmenter advances its sample timeline by `frame_samples` for
  /// each probability fed to [`push_probability`](Self::push_probability),
  /// so this must equal the frame size of the backend producing those
  /// probabilities — the value a [`VadBackend`] reports from
  /// [`frame_samples`](VadBackend::frame_samples). [`detect_speech_with`]
  /// applies it automatically; call it directly only when driving
  /// [`push_probability`](Self::push_probability) with a custom backend
  /// whose frame size differs from the sample rate's model chunk size.
  ///
  /// [`set_sample_rate`](Self::set_sample_rate) resets the frame size to
  /// the rate's model chunk size, so apply this after any rate change.
  ///
  /// # Panics
  ///
  /// Panics if `frame_samples` is zero.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_frame_samples(&mut self, frame_samples: usize) {
    assert!(frame_samples != 0, "frame_samples must be non-zero");
    self.frame_samples = frame_samples as u64;
  }

  /// Returns whether the segmenter is currently active (i.e., has an ongoing speech segment).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn is_active(&self) -> bool {
    self.active_start.is_some()
  }

  /// Reset the segmenter's internal state: the in-flight segment
  /// tracker (active start, tentative end, force-split bookkeeping),
  /// the running sample counter, and any segments queued for
  /// `push_samples(&[])` drain.
  ///
  /// This does not touch the `StreamState` buffer of un-chunked PCM
  /// (available with the `onnx` feature) — that lives on the stream, not
  /// the segmenter — so callers that reuse a stream for a new logical
  /// recording should also call `StreamState::reset` (or construct a
  /// fresh `StreamState`).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn reset(&mut self) {
    self.current_sample = 0;
    self.active_start = None;
    self.active_raw_start = None;
    self.tentative_end = None;
    self.max_split_end = None;
    self.next_start = None;
    self.pending_segments.clear();
  }

  /// Number of segments currently queued for drain via `push_samples(&[])`.
  ///
  /// Always `0` after a `push_samples` or `finish_stream` call that
  /// returned `Ok(None)`. Useful for tests that want to assert the
  /// caller has drained everything before tearing down a stream.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn pending_segment_count(&self) -> usize {
    self.pending_segments.len()
  }

  /// Consume one probability for one Silero frame.
  ///
  /// Returns `Some(segment)` only when a speech segment can be closed
  /// with the currently available evidence.
  pub fn push_probability(&mut self, probability: f32) -> Option<SpeechSegment> {
    let frame_samples = self.frame_samples;
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
        self.active_raw_start = Some(frame_start);
        return None;
      }
    }

    let start = self.active_start?;
    let raw_start = self.active_raw_start?;
    if let Some(max_speech_samples) = self
      .options
      .max_speech_samples_for_frame(self.frame_samples)
      && frame_start.saturating_sub(raw_start) > max_speech_samples
    {
      return self.split_at_max_duration(frame_start, probability);
    }

    if probability >= self.options.end_threshold() {
      return None;
    }

    // Silence-counter is evaluated against `frame_start` (the start sample
    // of the current frame), not `current_sample` (which is already the
    // *end* of the current frame). This matches upstream Python
    // `silero-vad`'s `sil_dur_now = cur_sample - temp_end` semantics,
    // where `cur_sample` is read BEFORE the model consumes the current
    // window. Without this, the comparator fires one frame early — a
    // 4-frame (128 ms) silence dip would close a segment at default
    // `min_silence_duration_ms = 100`, where Python tolerates it and
    // closes after 5 consecutive low-probability frames. See the parity
    // harness in `tests/parity/` and the v0.3.0 CHANGELOG entry.
    let silence_start = *self.tentative_end.get_or_insert(frame_start);
    let silence_samples = frame_start.saturating_sub(silence_start);
    if silence_samples > self.options.min_silence_at_max_speech_samples() {
      self.max_split_end = Some(silence_start);
    }
    if silence_samples < self.options.min_silence_samples() {
      return None;
    }

    self.clear_segment_memory();
    self.build_segment(start, raw_start, silence_start)
  }

  /// Feed PCM samples into one stream and return the next available
  /// closed segment.
  ///
  /// Returns `Ok(Some(segment))` when a segment is ready, `Ok(None)`
  /// when none is available yet. Pass an empty slice (`&[]`) to drain
  /// any segments still buffered from a previous call without feeding
  /// new audio — useful when a single push closed more than one
  /// segment (rare but possible at force-split).
  #[cfg(feature = "onnx")]
  #[cfg_attr(docsrs, doc(cfg(feature = "onnx")))]
  pub fn push_samples(
    &mut self,
    session: &mut Session,
    stream: &mut StreamState,
    samples: &[f32],
  ) -> Result<Option<SpeechSegment>> {
    self.ensure_sample_rate(stream.sample_rate())?;
    if !samples.is_empty() {
      // `Session::process_stream` is atomic: on inference failure it
      // restores `StreamState` to its pre-call snapshot and clears
      // its scratch. So the segmenter only needs to advance when the
      // call succeeds — partial-progress reconciliation is the
      // session's responsibility.
      let probabilities = session.process_stream(stream, samples)?;
      for &probability in probabilities {
        if let Some(segment) = self.push_probability(probability) {
          self.pending_segments.push_back(segment);
        }
      }
    }
    Ok(self.pending_segments.pop_front())
  }

  /// Zero-pad and process any remaining partial frame for a stream.
  ///
  /// If the flushed frame confirms the end of an active segment, the
  /// resulting segment is appended to the pending-segment queue.
  /// This call then pops and returns the **front** of that queue —
  /// so if earlier `push_samples` calls queued segments that the
  /// caller hasn't drained yet, those come out first, in order,
  /// before the flush-produced segment.
  ///
  /// Returns `Ok(None)` only when the queue is empty after the flush.
  /// Drain the rest of the queue with `push_samples(&[])` when this
  /// returns a segment, in case the flush plus prior pushes left
  /// more than one waiting.
  #[cfg(feature = "onnx")]
  #[cfg_attr(docsrs, doc(cfg(feature = "onnx")))]
  pub fn flush_stream(
    &mut self,
    session: &mut Session,
    stream: &mut StreamState,
  ) -> Result<Option<SpeechSegment>> {
    self.ensure_sample_rate(stream.sample_rate())?;
    if let Some(probability) = session.flush_stream(stream)?
      && let Some(segment) = self.push_probability(probability)
    {
      self.pending_segments.push_back(segment);
    }
    Ok(self.pending_segments.pop_front())
  }

  /// Compute the trailing open segment (if any) without resetting.
  /// Helper for `finish` and `finish_stream`.
  fn take_trailing(&self) -> Option<SpeechSegment> {
    let start = self.active_start?;
    let raw_start = self.active_raw_start?;
    let end = self.current_sample;
    if end.saturating_sub(raw_start) < self.options.min_speech_samples() {
      None
    } else {
      Some(SpeechSegment::new(start, end, self.sample_rate()))
    }
  }

  /// Finish the current stream and return the next available segment.
  ///
  /// Enqueues the trailing open segment (if any) onto the
  /// `pending_segments` queue, then pops and returns the head of
  /// that queue. This preserves the order of any segments that an
  /// earlier `push_samples` queued but the caller hasn't drained yet
  /// (the rare force-split case): they come out before the trailing
  /// segment.
  ///
  /// The in-flight segment tracker is cleared so `is_active()` is
  /// `false` afterwards and a follow-up `finish()` / `finish_stream()`
  /// can't re-emit the same trailing segment. The pending-segment
  /// queue is left intact so subsequent `push_samples(&[])` calls
  /// drain the rest; call [`Self::reset`] explicitly when starting a
  /// new stream.
  ///
  /// This does **not** flush the model tail — use `finish_stream`
  /// (available with the `onnx` feature) for the combined "flush model
  /// tail + close trailing segment" end-of-stream operation.
  pub fn finish(&mut self) -> Option<SpeechSegment> {
    if let Some(trailing) = self.take_trailing() {
      self.pending_segments.push_back(trailing);
    }
    self.clear_segment_memory();
    self.pending_segments.pop_front()
  }

  /// Convenience for end-of-stream handling: flush the model tail,
  /// close any trailing open segment, and return the next available
  /// segment from the resulting queue.
  ///
  /// Drain additional buffered segments with `push_samples(&[])` after
  /// this call, in case flush + close produced more than one segment.
  /// The in-flight segment tracker is cleared once the trailing
  /// segment has been enqueued so `is_active()` returns `false` and a
  /// follow-up `finish()` / `finish_stream()` can't re-emit the same
  /// segment. The pending-segment queue is left intact so the drain
  /// works; call [`Self::reset`] explicitly when starting a new
  /// stream.
  #[cfg(feature = "onnx")]
  #[cfg_attr(docsrs, doc(cfg(feature = "onnx")))]
  pub fn finish_stream(
    &mut self,
    session: &mut Session,
    stream: &mut StreamState,
  ) -> Result<Option<SpeechSegment>> {
    self.ensure_sample_rate(stream.sample_rate())?;
    if let Some(probability) = session.flush_stream(stream)?
      && let Some(segment) = self.push_probability(probability)
    {
      self.pending_segments.push_back(segment);
    }
    if let Some(trailing) = self.take_trailing() {
      self.pending_segments.push_back(trailing);
    }
    // Clear the in-flight segment tracker so `is_active()` reflects
    // end-of-stream and a follow-up finish call can't re-emit the
    // trailing segment. Keep `pending_segments` intact for drain.
    self.clear_segment_memory();
    Ok(self.pending_segments.pop_front())
  }

  #[cfg(feature = "onnx")]
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
    let raw_start = self.active_raw_start?;
    let raw_end = self.max_split_end.unwrap_or(frame_start);
    let segment = self.build_segment(start, raw_start, raw_end);

    let next_raw_start = if let Some(next_start) = self.next_start.filter(|next| *next >= raw_end) {
      self.active_start = Some(next_start.saturating_sub(self.options.speech_pad_samples()));
      Some(next_start)
    } else if self.max_split_end.is_none() && probability >= self.options.start_threshold() {
      self.active_start = Some(frame_start.saturating_sub(self.options.speech_pad_samples()));
      Some(frame_start)
    } else {
      self.active_start = None;
      None
    };
    self.active_raw_start = next_raw_start;
    self.clear_split_tracking();

    segment
  }

  fn build_segment(&self, start: u64, raw_start: u64, raw_end: u64) -> Option<SpeechSegment> {
    let end_sample = raw_end
      .saturating_add(self.options.speech_pad_samples())
      .min(self.current_sample);
    if raw_end.saturating_sub(raw_start) < self.options.min_speech_samples() {
      None
    } else {
      Some(SpeechSegment::new(start, end_sample, self.sample_rate()))
    }
  }

  fn clear_segment_memory(&mut self) {
    self.active_start = None;
    self.active_raw_start = None;
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

/// Convenience helper for one-shot offline detection on a full buffer
/// using the bundled ONNX backend.
///
/// See [`detect_speech_with`] for the backend-agnostic counterpart.
#[cfg(feature = "onnx")]
#[cfg_attr(docsrs, doc(cfg(feature = "onnx")))]
pub fn detect_speech(
  session: &mut Session,
  samples: &[f32],
  config: SpeechOptions,
) -> Result<Vec<SpeechSegment>> {
  let mut stream = StreamState::new(config.sample_rate());
  let mut segmenter = SpeechSegmenter::new(config);
  let mut segments = Vec::new();
  if let Some(segment) = segmenter.push_samples(session, &mut stream, samples)? {
    segments.push(segment);
    while let Some(more) = segmenter.push_samples(session, &mut stream, &[])? {
      segments.push(more);
    }
  }
  if let Some(segment) = segmenter.finish_stream(session, &mut stream)? {
    segments.push(segment);
    while let Some(more) = segmenter.push_samples(session, &mut stream, &[])? {
      segments.push(more);
    }
  }
  Ok(segments)
}

/// One-shot offline speech detection over any [`VadBackend`].
///
/// The backend-agnostic counterpart to `detect_speech` (the bundled
/// `onnx` helper): it chunks `samples` into
/// [`frame_samples`](VadBackend::frame_samples)-sized
/// frames, runs the backend once per frame, and applies the same
/// segmentation rules as [`SpeechSegmenter`]. A trailing partial frame
/// is zero-padded and flushed, matching `detect_speech`'s end-of-stream
/// behavior. The backend is *not* [`reset`](VadBackend::reset): pass a
/// freshly constructed or reset backend to start a new stream.
///
/// # Sample rate
///
/// The backend is authoritative for its own stream: the segmenter's
/// duration thresholds and emitted [`SpeechSegment`] stamps are taken
/// from [`backend.sample_rate()`](VadBackend::sample_rate), overriding
/// whatever `sample_rate` the passed `options` carried. Configure the
/// rate on the backend, not on `options`, when driving this helper.
///
/// # Errors
///
/// Returns the backend's error, bridged into [`Error`](crate::Error), if
/// any frame's inference fails.
///
/// # Panics
///
/// Panics if the backend reports a zero
/// [`frame_samples`](VadBackend::frame_samples).
pub fn detect_speech_with<B: VadBackend>(
  backend: &mut B,
  samples: &[f32],
  options: SpeechOptions,
) -> Result<Vec<SpeechSegment>> {
  let frame = backend.frame_samples();
  assert!(frame != 0, "VadBackend::frame_samples() must be non-zero");
  let mut segmenter = SpeechSegmenter::new(options);
  // The backend owns its stream's sample rate: align the segmenter's
  // duration conversions and segment stamps to it, overriding the rate
  // the passed options carried. `set_sample_rate` also resets the frame
  // geometry to that rate's model chunk, so re-apply the backend's frame
  // size afterward.
  segmenter.set_sample_rate(backend.sample_rate());
  segmenter.set_frame_samples(frame);
  let mut segments = Vec::new();

  let mut offset = 0;
  while offset + frame <= samples.len() {
    let probability = backend
      .predict(&samples[offset..offset + frame])
      .map_err(Into::into)?;
    if let Some(segment) = segmenter.push_probability(probability) {
      segments.push(segment);
    }
    offset += frame;
  }

  // Zero-pad and flush the trailing partial frame, mirroring
  // `Session::flush_stream` at end-of-stream.
  if offset < samples.len() {
    let mut tail = vec![0.0; frame];
    tail[..samples.len() - offset].copy_from_slice(&samples[offset..]);
    let probability = backend.predict(&tail).map_err(Into::into)?;
    if let Some(segment) = segmenter.push_probability(probability) {
      segments.push(segment);
    }
  }

  if let Some(segment) = segmenter.finish() {
    segments.push(segment);
  }
  Ok(segments)
}

#[cfg(test)]
mod tests {
  use std::time::Duration;

  use crate::{SampleRate, SpeechOptions, VadBackend};

  use super::{SpeechSegment, SpeechSegmenter, detect_speech_with};

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
    let mut segmenter = SpeechSegmenter::new(config.clone());
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
    let mut segmenter = SpeechSegmenter::new(config.clone());
    let mut probabilities = vec![0.9; frame_count(64, SampleRate::Rate16k)];
    probabilities.extend(vec![0.0; frame_count(160, SampleRate::Rate16k)]);
    let segments = collect(&mut segmenter, &probabilities);
    assert!(segments.is_empty());
  }

  #[test]
  fn middle_band_frames_do_not_reset_tentative_end() {
    // Verifies that mid-band probabilities (between the end_threshold and
    // start_threshold, e.g. `0.4` against the default `0.5` start) do NOT
    // reset the silence accumulator — they're treated as "not yet
    // confirmed speech".
    //
    // Updated 0.3.0: post the silence-counter off-by-one fix, the segment
    // closes after FIVE consecutive low-or-mid-band frames at the default
    // `min_silence_duration_ms = 100` (1600 samples / 512 per frame =
    // 3.125 → 4 prior frames + the close-firing 5th frame), matching
    // upstream Python silero-vad. The pre-0.3.0 crate closed after FOUR
    // frames (one frame too eager). See `tests/parity/README.md` and the
    // 0.3.0 CHANGELOG entry for the full derivation.
    let config = SpeechOptions::default()
      .with_min_speech_duration(Duration::ZERO)
      .with_speech_pad(Duration::ZERO)
      .with_min_silence_duration(Duration::from_millis(100));
    let mut segmenter = SpeechSegmenter::new(config);

    let mut probabilities = vec![0.9; 4];
    // Five low/mid frames so the segment closes via push_probability.
    // The mid-band 0.4 frame in the middle must NOT reset the silence
    // accumulator — that's the actual property under test.
    probabilities.extend([0.0, 0.4, 0.0, 0.0, 0.0]);
    probabilities.extend(vec![0.9; 4]);

    let segments = collect(&mut segmenter, &probabilities);
    assert_eq!(segments.len(), 2);
    assert_eq!(segments[0].start_sample(), 0);
    assert_eq!(segments[0].end_sample(), 2_048);
    // Segment two starts on the first speech frame after the closed
    // silence (4 high + 5 silence = frame index 9, sample 4_608).
    assert_eq!(segments[1].start_sample(), 4_608);
  }

  #[test]
  fn min_speech_duration_is_checked_before_padding() {
    // A speech burst of 6 frames * 32 ms = 192 ms is shorter than the
    // default `min_speech_duration_ms = 250`, so the segment that the
    // trailing silence closes must be dropped — `min_speech` is checked
    // against the raw speech window (raw_end - raw_start), not against
    // the padded boundaries.
    //
    // Updated 0.3.0: post the silence-counter off-by-one fix, push-based
    // close requires FIVE consecutive low-probability frames at the
    // default `min_silence_duration_ms = 100` (was 4 pre-0.3.0). Trailing
    // silence is extended from 4 to 5 frames so the close still fires
    // via `push_probability` — otherwise `finish()` would emit the
    // burst-plus-trailing-silence as a single trailing segment that
    // satisfies the 250 ms duration check, which is a different (and
    // correct, but separate) behaviour. See `tests/parity/README.md`
    // and the 0.3.0 CHANGELOG entry.
    let config = SpeechOptions::default();
    let mut segmenter = SpeechSegmenter::new(config);

    let mut probabilities = vec![0.0; 4];
    probabilities.extend(vec![0.9; 6]);
    probabilities.extend(vec![0.0; 5]);

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
      .with_min_speech_duration(Duration::ZERO)
      .with_speech_pad(Duration::ZERO)
      .with_max_speech_duration(Duration::from_millis(160));
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
      .with_min_speech_duration(Duration::ZERO)
      .with_speech_pad(Duration::ZERO)
      .with_min_silence_duration(Duration::from_millis(300))
      .with_min_silence_at_max_speech(Duration::from_millis(64))
      .with_max_speech_duration(Duration::from_millis(256));
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
      .with_min_speech_duration(Duration::ZERO)
      .with_speech_pad(Duration::ZERO)
      .with_min_silence_duration(Duration::from_millis(10_000))
      .with_min_silence_at_max_speech(Duration::from_millis(64))
      .with_max_speech_duration(Duration::from_millis(512));
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
    // Updated 0.3.0: max_speech_duration bumped from 224 ms to 256 ms so
    // the max-speech split fires one frame later, after `max_split_end`
    // has been recorded by the silence-counter logic. With the
    // off-by-one fix to that logic, `max_split_end` is now set on the
    // 4th low-probability frame instead of the 3rd, so the test's
    // pre-existing 224 ms ceiling would split at sample 3_584 with
    // `max_split_end == None` (falling back to `frame_start` and
    // closing at sample 3_584 instead of at the recorded silence
    // boundary 2_048). Bumping the ceiling preserves the property under
    // test — that a force-split during silence closes at the silence
    // boundary, not at the current frame, and does NOT restart a new
    // segment afterwards. See `tests/parity/README.md` and the 0.3.0
    // CHANGELOG entry.
    let config = SpeechOptions::default()
      .with_min_speech_duration(Duration::ZERO)
      .with_speech_pad(Duration::ZERO)
      .with_min_silence_duration(Duration::from_millis(10_000))
      .with_min_silence_at_max_speech(Duration::from_millis(64))
      .with_max_speech_duration(Duration::from_millis(256));
    let mut segmenter = SpeechSegmenter::new(config);

    let mut probabilities = vec![0.9; 4];
    probabilities.extend(vec![0.0; 8]);

    let segments = collect(&mut segmenter, &probabilities);
    assert_eq!(segments.len(), 1);
    assert_eq!(segments[0].start_sample(), 0);
    assert_eq!(segments[0].end_sample(), 2_048);
  }

  #[test]
  fn four_frame_silence_dip_does_not_close_segment_at_default_min_silence() {
    // Pinned in 0.3.0 as a regression guard for the silence-counter
    // off-by-one fix.
    //
    // At the default `min_silence_duration_ms = 100` (1600 samples at
    // 16 kHz) and the default 32 ms / 512-sample frame, upstream Python
    // `silero-vad` (`get_speech_timestamps`) closes a segment after
    // FIVE consecutive low-probability frames — `sil_dur_now =
    // cur_sample - temp_end` is evaluated BEFORE the current frame is
    // consumed, so the comparator sees `(k-1) * 512` on the k-th
    // low-prob frame and only crosses the 1600-sample threshold at
    // k = 5.
    //
    // Pre-0.3.0 the silero crate evaluated the same counter AFTER the
    // current frame was added to `current_sample`, so it saw `k * 512`
    // and closed at k = 4. A 4-frame (128 ms) silence dip would
    // therefore split a segment in the crate but be tolerated by Python.
    //
    // This test pins the post-fix behaviour: a 4-frame silence dip must
    // be tolerated. The 30-frame speech runs ensure both halves
    // individually clear `min_speech_duration_ms = 250` (8 frames),
    // so neither would be dropped by the min-speech filter if the
    // segment did split.
    //
    // See `tests/parity/README.md` "Off-by-one silence threshold finding"
    // and the 0.3.0 CHANGELOG entry for the motivation.
    let config = SpeechOptions::default();
    let mut segmenter = SpeechSegmenter::new(config.clone());

    let mut probabilities = vec![1.0; 30];
    probabilities.extend(vec![0.0; 4]);
    probabilities.extend(vec![1.0; 30]);

    let segments = collect(&mut segmenter, &probabilities);
    assert_eq!(
      segments.len(),
      1,
      "4-frame silence dip must be tolerated at default min_silence_duration_ms = 100; \
       got {} segments",
      segments.len()
    );
    // Sanity: the (one) segment must start at 0 (the start-pad
    // saturates against the timeline's zero) and span the full
    // 30 + 4 + 30 = 64 frame window — at 512 samples / frame, that
    // ends at 32_768.
    assert_eq!(segments[0].start_sample(), 0);
    assert_eq!(segments[0].end_sample(), 32_768);
  }

  #[test]
  fn five_frame_silence_dip_closes_segment_at_default_min_silence() {
    // Companion to `four_frame_silence_dip_does_not_close_segment_*`.
    // Pinned in 0.3.0: at the same defaults, FIVE consecutive low-prob
    // frames must close the segment — matching upstream Python
    // silero-vad's `sil_dur_now >= 1600` firing on the 5th frame.
    let config = SpeechOptions::default();
    let mut segmenter = SpeechSegmenter::new(config);

    let mut probabilities = vec![1.0; 30];
    probabilities.extend(vec![0.0; 5]);
    probabilities.extend(vec![1.0; 30]);

    let segments = collect(&mut segmenter, &probabilities);
    assert_eq!(
      segments.len(),
      2,
      "5-frame silence dip must close the segment at default \
       min_silence_duration_ms = 100; got {} segments",
      segments.len()
    );
  }

  #[test]
  fn force_split_applies_speech_pad_to_split_boundaries() {
    let config = SpeechOptions::default()
      .with_min_speech_duration(Duration::ZERO)
      .with_speech_pad(Duration::from_millis(32))
      .with_min_silence_duration(Duration::from_millis(10_000))
      .with_min_silence_at_max_speech(Duration::from_millis(64))
      .with_max_speech_duration(Duration::from_millis(512));
    let mut segmenter = SpeechSegmenter::new(config);

    let mut probabilities = vec![0.9; 4];
    probabilities.extend(vec![0.0; 4]);
    probabilities.extend(vec![0.9; 8]);

    let segments = collect(&mut segmenter, &probabilities);
    assert_eq!(segments[0].end_sample(), 2_560);
    assert_eq!(segments[1].start_sample(), 3_584);
  }

  #[test]
  fn finish_preserves_undrained_queued_segments() {
    // Pin in 0.4.0 (codex round-3 finding): `push_samples` can queue
    // multiple segments per call (rare but possible — a long buffer
    // with a force-split + close in one push). The previous `finish()`
    // implementation called `reset()`, which cleared the queue and
    // silently lost any segments the caller hadn't popped yet.
    //
    // The new contract: `finish()` enqueues the trailing segment (if
    // any) at the back of the queue and pops the front, so undrained
    // segments come out in order before the trailing one.
    let config = SpeechOptions::default();
    let mut segmenter = SpeechSegmenter::new(config);

    // Simulate the post-`push_samples` state where two segments
    // closed in one call but the caller only popped the first: stage
    // two segments in the queue directly via the (private) field.
    let queued_a = SpeechSegment::new(0, 1_000, segmenter.sample_rate());
    let queued_b = SpeechSegment::new(2_000, 3_000, segmenter.sample_rate());
    segmenter.pending_segments.push_back(queued_a);
    segmenter.pending_segments.push_back(queued_b);

    // First finish(): must return the head of the queue, NOT silently
    // drop the rest.
    assert_eq!(segmenter.finish(), Some(queued_a));
    assert_eq!(segmenter.pending_segment_count(), 1);

    // Second finish(): must drain the next queued segment.
    assert_eq!(segmenter.finish(), Some(queued_b));
    assert_eq!(segmenter.pending_segment_count(), 0);

    // Queue exhausted, no trailing → None.
    assert_eq!(segmenter.finish(), None);

    // Active state cleared by finish() so a subsequent push could
    // start a fresh segment cleanly.
    assert!(!segmenter.is_active());
  }

  // ── Backend seam: hermetic mock (no ORT) ──────────────────────────

  /// A backend error distinct from `crate::Error`, exercising the
  /// associated-error bridge an out-of-tree backend would use.
  #[derive(Debug)]
  struct MockError(&'static str);

  impl std::fmt::Display for MockError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
      f.write_str(self.0)
    }
  }

  impl std::error::Error for MockError {}

  impl From<MockError> for crate::Error {
    fn from(error: MockError) -> Self {
      crate::Error::Backend(Box::new(error))
    }
  }

  /// A `VadBackend` that returns canned probabilities — one per frame —
  /// at a caller-declared frame geometry. It authors no detection logic;
  /// it exists to drive the segmenter's frame math and the error bridge
  /// without ORT.
  struct MockBackend {
    frame_samples: usize,
    sample_rate: SampleRate,
    probabilities: Vec<f32>,
    cursor: usize,
    fail_at: Option<usize>,
  }

  impl MockBackend {
    fn new(frame_samples: usize, probabilities: Vec<f32>) -> Self {
      Self {
        frame_samples,
        sample_rate: SampleRate::Rate16k,
        probabilities,
        cursor: 0,
        fail_at: None,
      }
    }

    fn with_sample_rate(mut self, sample_rate: SampleRate) -> Self {
      self.sample_rate = sample_rate;
      self
    }
  }

  impl VadBackend for MockBackend {
    type Error = MockError;

    fn frame_samples(&self) -> usize {
      self.frame_samples
    }

    fn sample_rate(&self) -> SampleRate {
      self.sample_rate
    }

    fn predict(&mut self, frame: &[f32]) -> Result<f32, MockError> {
      assert_eq!(
        frame.len(),
        self.frame_samples,
        "backend must be handed exactly frame_samples per frame"
      );
      if self.fail_at == Some(self.cursor) {
        return Err(MockError("mock predict failure"));
      }
      let probability = self.probabilities.get(self.cursor).copied().unwrap_or(0.0);
      self.cursor += 1;
      Ok(probability)
    }

    fn reset(&mut self) {
      self.cursor = 0;
    }
  }

  #[test]
  fn set_frame_samples_overrides_the_sample_rate_default() {
    let mut segmenter = SpeechSegmenter::new(SpeechOptions::default());
    // The 16 kHz model chunk is the default frame geometry.
    assert_eq!(segmenter.frame_samples(), 512);
    segmenter.set_frame_samples(4096);
    assert_eq!(segmenter.frame_samples(), 4096);
  }

  #[test]
  fn detect_speech_with_uses_backend_sample_rate_not_options() {
    // Regression (backend-seam High): the backend is authoritative for
    // its stream's sample rate. An 8 kHz / 256-sample backend feeding
    // 320 ms of speech (ten 0.9 frames over 2_560 samples) must produce
    // one segment — 320 ms clears the 250 ms minimum at 8 kHz (2_000
    // samples) — stamped 8 kHz, even though the passed options carry the
    // default 16 kHz. Before the fix `detect_speech_with` built the
    // segmenter straight from the 16 kHz options: it converted 250 ms to
    // 4_000 samples, dropped the 2_560-sample run, and would have stamped
    // any segment 16 kHz. Mutation: drop the `set_sample_rate` derivation
    // in `detect_speech_with` → zero segments (and a 16 kHz stamp) → red.
    let mut backend = MockBackend::new(256, vec![0.9; 10]).with_sample_rate(SampleRate::Rate8k);
    let samples = vec![0.0_f32; 10 * 256];
    let segments =
      detect_speech_with(&mut backend, &samples, SpeechOptions::default()).expect("detect");
    assert_eq!(
      segments.len(),
      1,
      "320 ms of speech at the backend's 8 kHz must yield one segment"
    );
    assert_eq!(
      segments[0].sample_rate(),
      SampleRate::Rate8k,
      "segment must be stamped with the backend's rate"
    );
  }

  #[test]
  fn mock_geometry_closes_after_two_256ms_low_frames() {
    // A backend declaring 4096-sample frames makes each frame 256 ms at
    // 16 kHz. The default `min_silence_duration_ms = 100` is 1600
    // samples. Because the silence counter is measured BEFORE the
    // current frame is consumed, the FIRST low frame only establishes
    // the silence start (counter 0) and the SECOND low frame sees a full
    // 4096-sample (256 ms) gap — which already exceeds 1600 — so the
    // segment closes on the second low frame. This is the above-the-
    // threshold side of the duration→frame rounding, and every boundary
    // lands on a 4096-sample multiple.
    let mut backend = MockBackend::new(4096, vec![0.9, 0.9, 0.9, 0.0, 0.0]);
    let samples = vec![0.0_f32; 5 * 4096];
    let segments =
      detect_speech_with(&mut backend, &samples, SpeechOptions::default()).expect("detect");

    assert_eq!(
      segments.len(),
      1,
      "two 256 ms low frames must close one segment"
    );
    assert_eq!(segments[0].start_sample(), 0);
    // raw_end = 3 * 4096 (silence start), + 30 ms speech_pad (480).
    assert_eq!(segments[0].end_sample(), 3 * 4096 + 480);
    // Every frame handed to the backend was consumed at 4096 samples.
    assert_eq!(backend.cursor, 5);
  }

  #[test]
  fn mock_geometry_holds_open_through_one_256ms_low_frame() {
    // The below-the-threshold side: a SINGLE 256 ms low frame only
    // establishes the silence start (counter 0 < 1600), so no segment
    // closes mid-stream. The open segment is emitted by the end-of-
    // stream `finish`, spanning to the raw current sample — a 4096-
    // sample multiple with no trailing pad. Hardcoding a 512-sample
    // frame here would advance the timeline too slowly to satisfy the
    // 250 ms (4000-sample) minimum-speech gate and drop the segment.
    let mut backend = MockBackend::new(4096, vec![0.9, 0.9, 0.9, 0.0]);
    let samples = vec![0.0_f32; 4 * 4096];
    let segments =
      detect_speech_with(&mut backend, &samples, SpeechOptions::default()).expect("detect");

    assert_eq!(
      segments.len(),
      1,
      "one 256 ms low frame must not close the segment"
    );
    assert_eq!(segments[0].start_sample(), 0);
    assert_eq!(segments[0].end_sample(), 4 * 4096);
  }

  #[test]
  fn mock_geometry_max_speech_lookahead_is_frame_aware() {
    // Regression (backend-seam Medium): the max-speech force-split
    // lookahead must subtract the ACTIVE frame size, not the sample rate's
    // model chunk. A 4096-sample backend at 16 kHz with a 1 s max-speech
    // ceiling (speech_pad 0) has a frame-aware threshold of
    // 16_000 − 4_096 = 11_904, so the first frame_start past it is 12_288
    // (768 ms) and the split lands there. The old chunk-based threshold
    // (16_000 − 512 = 15_488) split one frame later, at frame_start
    // 16_384 (1.024 s) — overshooting the configured 1 s maximum. Mutation:
    // revert the lookahead to `chunk_samples()` → the split moves to
    // 16_384 → red.
    let mut backend = MockBackend::new(4096, vec![0.9; 6]);
    let samples = vec![0.0_f32; 6 * 4096];
    let options = SpeechOptions::default()
      .with_min_speech_duration(Duration::ZERO)
      .with_speech_pad(Duration::ZERO)
      .with_max_speech_duration(Duration::from_millis(1_000));
    let segments = detect_speech_with(&mut backend, &samples, options).expect("detect");

    assert_eq!(segments[0].start_sample(), 0);
    assert_eq!(
      segments[0].end_sample(),
      12_288,
      "max-speech split must land at the frame-aware 12_288, not the \
       chunk-based overshoot 16_384"
    );
  }

  #[test]
  fn mock_backend_error_bridges_through_backend_variant() {
    // The associated `VadBackend::Error` (a foreign type here) must
    // reach the caller through the transparent `Error::Backend` variant,
    // delegating its `Display` to the wrapped error.
    let mut backend = MockBackend {
      frame_samples: 4096,
      sample_rate: SampleRate::Rate16k,
      probabilities: vec![0.9, 0.9, 0.9],
      cursor: 0,
      fail_at: Some(1),
    };
    let samples = vec![0.0_f32; 3 * 4096];
    let error = detect_speech_with(&mut backend, &samples, SpeechOptions::default())
      .expect_err("backend failure must propagate");
    assert!(
      matches!(error, crate::Error::Backend(_)),
      "backend error must bridge through Error::Backend, got {error:?}"
    );
    assert_eq!(error.to_string(), "mock predict failure");
  }
}
