use crate::{
  Result, SampleRate, Session, SpeechOptions, SpeechSegment, SpeechSegmenter, StreamState,
};

/// ONNX streaming-driver extension for [`SpeechSegmenter`].
///
/// The backend-agnostic [`SpeechSegmenter`] lives in the [`zuoer`] core;
/// these methods drive it from the bundled ONNX [`Session`] +
/// [`StreamState`]. They are an extension trait rather than inherent
/// methods because `SpeechSegmenter` is a foreign type — bring the trait
/// into scope (`use silero::SpeechSegmenterExt`) to call them.
///
/// Each method is a thin wrapper over the segmenter's sans-I/O seam
/// ([`SpeechSegmenter::push_probabilities`] / [`pop_pending`] /
/// [`finish`]): it runs the session to turn PCM into frame probabilities,
/// feeds them to the segmenter, and drains the closed segments.
///
/// [`pop_pending`]: SpeechSegmenter::pop_pending
/// [`finish`]: SpeechSegmenter::finish
pub trait SpeechSegmenterExt {
  /// Feed PCM samples into one stream and return the next available
  /// closed segment.
  ///
  /// Returns `Ok(Some(segment))` when a segment is ready, `Ok(None)`
  /// when none is available yet. Pass an empty slice (`&[]`) to drain
  /// any segments still buffered from a previous call without feeding
  /// new audio — useful when a single push closed more than one
  /// segment (rare but possible at force-split).
  fn push_samples(
    &mut self,
    session: &mut Session,
    stream: &mut StreamState,
    samples: &[f32],
  ) -> Result<Option<SpeechSegment>>;

  /// Zero-pad and process any remaining partial frame for a stream.
  ///
  /// If the flushed frame confirms the end of an active segment, the
  /// resulting segment is appended to the pending-segment queue.
  /// This call then pops and returns the **front** of that queue — so if
  /// earlier `push_samples` calls queued segments that the caller hasn't
  /// drained yet, those come out first, in order, before the
  /// flush-produced segment.
  ///
  /// Returns `Ok(None)` only when the queue is empty after the flush.
  fn flush_stream(
    &mut self,
    session: &mut Session,
    stream: &mut StreamState,
  ) -> Result<Option<SpeechSegment>>;

  /// Convenience for end-of-stream handling: flush the model tail,
  /// close any trailing open segment, and return the next available
  /// segment from the resulting queue.
  ///
  /// Drain additional buffered segments with `push_samples(&[])` after
  /// this call, in case flush + close produced more than one segment.
  /// The in-flight segment tracker is cleared once the trailing segment
  /// has been enqueued so `is_active()` returns `false` and a follow-up
  /// `finish_stream()` / `finish()` can't re-emit the same segment.
  fn finish_stream(
    &mut self,
    session: &mut Session,
    stream: &mut StreamState,
  ) -> Result<Option<SpeechSegment>>;
}

impl SpeechSegmenterExt for SpeechSegmenter {
  fn push_samples(
    &mut self,
    session: &mut Session,
    stream: &mut StreamState,
    samples: &[f32],
  ) -> Result<Option<SpeechSegment>> {
    ensure_sample_rate(self, stream.sample_rate())?;
    if !samples.is_empty() {
      // `Session::process_stream` is atomic: on inference failure it
      // restores `StreamState` to its pre-call snapshot and clears its
      // scratch. So the segmenter only needs to advance when the call
      // succeeds — partial-progress reconciliation is the session's
      // responsibility.
      let probabilities = session.process_stream(stream, samples)?;
      self.push_probabilities(probabilities);
    }
    Ok(self.pop_pending())
  }

  fn flush_stream(
    &mut self,
    session: &mut Session,
    stream: &mut StreamState,
  ) -> Result<Option<SpeechSegment>> {
    ensure_sample_rate(self, stream.sample_rate())?;
    if let Some(probability) = session.flush_stream(stream)? {
      self.push_probabilities(&[probability]);
    }
    Ok(self.pop_pending())
  }

  fn finish_stream(
    &mut self,
    session: &mut Session,
    stream: &mut StreamState,
  ) -> Result<Option<SpeechSegment>> {
    ensure_sample_rate(self, stream.sample_rate())?;
    if let Some(probability) = session.flush_stream(stream)? {
      self.push_probabilities(&[probability]);
    }
    Ok(self.finish())
  }
}

/// Verify a stream's sample rate matches the segmenter's timeline rate,
/// mirroring the pre-split inherent `ensure_sample_rate` check.
fn ensure_sample_rate(segmenter: &SpeechSegmenter, sample_rate: SampleRate) -> Result<()> {
  if segmenter.sample_rate() == sample_rate {
    Ok(())
  } else {
    Err(
      zuoer::Error::IncompatibleSampleRate {
        expected: segmenter.sample_rate().hz(),
        actual: sample_rate.hz(),
      }
      .into(),
    )
  }
}

/// Convenience helper for one-shot offline detection on a full buffer
/// using the bundled ONNX backend.
///
/// See [`detect_speech_with`](crate::detect_speech_with) for the
/// backend-agnostic counterpart.
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
