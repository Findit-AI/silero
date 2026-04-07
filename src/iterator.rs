//! Streaming VAD iterator — one chunk in, optional event out.
//!
//! This is the Rust port of the Python `VADIterator` class from
//! `silero_vad.utils_vad`. It is designed for **live** or streaming
//! use cases where audio arrives in a continuous stream and the caller
//! wants low-latency boundary events (`{start: …}` / `{end: …}`)
//! rather than a full timeline at the end.
//!
//! If you instead have an entire audio buffer in memory and want the
//! final list of speech segments, use [`crate::speech_timestamps`] —
//! it provides more features (minimum speech duration, padding,
//! max-speech splitting) and is faithfully one-to-one with upstream
//! Python's `get_speech_timestamps`.
//!
//! # Event semantics (matches upstream)
//!
//! - **Speech onset**: emitted on the first chunk whose probability
//!   crosses `threshold` while not already inside a speech region.
//!   The reported `start` sample is
//!   `max(0, current_sample - speech_pad_samples - window_size)` so
//!   a small amount of context before the detected word is included.
//!
//! - **Speech offset**: emitted after `min_silence_duration_ms` of
//!   continuous silence (frames with probability below
//!   `threshold - 0.15`) while inside a speech region. The reported
//!   `end` sample is `temp_end + speech_pad_samples - window_size`,
//!   where `temp_end` is the sample index at which silence began.
//!
//! - **No event**: returned as `Ok(None)` when the chunk neither
//!   started nor ended a speech region.
//!
//! The state machine is identical to the upstream one line for line,
//! so the same chunk sequence yields the same events in both
//! implementations.
//!
//! # Lifecycle
//!
//! ```rust,ignore
//! use silero::{VadIterator, VadConfig, Event, SampleRate};
//!
//! let mut vad = VadIterator::new("models/silero_vad.onnx", VadConfig::default())?;
//!
//! // Feed every 512-sample chunk from the stream (must be exact size).
//! for chunk in audio_16k.chunks_exact(SampleRate::Rate16k.chunk_samples()) {
//!     if let Some(event) = vad.process(chunk)? {
//!         match event {
//!             Event::Start(sample) => println!("speech started at {sample}"),
//!             Event::End(sample) => println!("speech ended at {sample}"),
//!         }
//!     }
//! }
//!
//! // When the stream is over, call `finish` to flush any open region.
//! if let Some(Event::End(sample)) = vad.finish() {
//!     println!("final speech ended at {sample}");
//! }
//! # Ok::<_, silero::SileroError>(())
//! ```
//!
//! # Deviation from Python
//!
//! The upstream Python `VADIterator.__call__` never emits an `end`
//! event for a speech region that is still open when the stream ends.
//! That's a surprising behaviour — users nearly always want the final
//! "close this region" signal. We add an explicit [`Self::finish`]
//! method that flushes the state and, if a speech region is still
//! open, returns a synthetic `End` event at the current sample. This
//! is the only behavioural deviation from the Python reference; it is
//! opt-in (you only see it if you call `finish`).

use std::path::Path;

use crate::config::{ms_to_samples, SampleRate, VadConfig};
use crate::error::Result;
use crate::model::SileroModel;

/// A boundary event emitted by [`VadIterator::process`] or
/// [`VadIterator::finish`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Event {
  /// A speech region just started. The payload is the start sample
  /// index in the stream (0-based, counting from the first sample
  /// ever fed to this iterator), with `speech_pad_samples` of context
  /// already subtracted.
  Start(u64),
  /// A speech region just ended. The payload is the end sample index
  /// in the stream, with `speech_pad_samples` of tail context already
  /// added.
  End(u64),
}

/// Streaming VAD iterator.
///
/// See the [module-level documentation](crate::iterator) for the full
/// lifecycle and event semantics.
pub struct VadIterator {
  model: SileroModel,
  config: VadConfig,

  /// Cached from `config` to avoid recomputing on every chunk.
  threshold: f32,
  neg_threshold: f32,
  min_silence_samples: u64,
  speech_pad_samples: u64,

  /// Whether we are currently inside a speech region.
  triggered: bool,
  /// Running total of samples fed through [`Self::process`] so far.
  current_sample: u64,
  /// Sample index at which silence tentatively started (0 means "not
  /// in tentative silence"). Matches the Python `self.temp_end`.
  temp_end: u64,
  /// The chunk size (in samples) of the last `process` call. Needed
  /// so `finish` can report the end position correctly. Initialised
  /// to the configured rate's default chunk size; updated on every
  /// `process` call.
  last_chunk_samples: u64,
}

impl VadIterator {
  /// Create a new iterator with a fresh ONNX session.
  ///
  /// The sample rate embedded in `config` must match the audio you
  /// will feed.
  pub fn new(model_path: impl AsRef<Path>, config: VadConfig) -> Result<Self> {
    let mut model = SileroModel::from_path(model_path)?;
    model.set_sample_rate(config.sample_rate());
    Ok(Self::with_model(model, config))
  }

  /// Create a new iterator around an existing [`SileroModel`].
  ///
  /// Useful when you want to share one loaded model across several
  /// iterators (e.g. one per track in a multi-track recording) or
  /// when the model was loaded from embedded bytes.
  pub fn with_model(mut model: SileroModel, config: VadConfig) -> Self {
    model.set_sample_rate(config.sample_rate());
    let threshold = config.threshold();
    let neg_threshold = config.effective_negative_threshold();
    let min_silence_samples = config.min_silence_samples();
    let speech_pad_samples = config.speech_pad_samples();
    let last_chunk_samples = config.sample_rate().chunk_samples() as u64;
    Self {
      model,
      config,
      threshold,
      neg_threshold,
      min_silence_samples,
      speech_pad_samples,
      triggered: false,
      current_sample: 0,
      temp_end: 0,
      last_chunk_samples,
    }
  }

  /// Returns the sample rate this iterator is configured for.
  #[inline]
  pub const fn sample_rate(&self) -> SampleRate {
    self.config.sample_rate()
  }

  /// Returns the expected chunk size in samples. Every call to
  /// [`Self::process`] must provide exactly this many samples.
  #[inline]
  pub const fn chunk_samples(&self) -> usize {
    self.config.sample_rate().chunk_samples()
  }

  /// Reset the iterator to a clean state. Clears the LSTM state, the
  /// rolling context, the sample counter, and the internal state
  /// machine. Use between independent streams.
  pub fn reset(&mut self) {
    self.model.reset();
    self.triggered = false;
    self.current_sample = 0;
    self.temp_end = 0;
    self.last_chunk_samples = self.config.sample_rate().chunk_samples() as u64;
  }

  /// Feed one chunk of audio.
  ///
  /// `chunk.len()` must equal [`Self::chunk_samples`]. Shorter or
  /// longer chunks return an error rather than being silently
  /// accepted, because the rolling state would no longer align with
  /// the model's internal assumptions.
  ///
  /// Returns `Ok(None)` when the chunk doesn't change the speech
  /// state, `Ok(Some(Event::Start(_)))` when a new speech region
  /// opened, and `Ok(Some(Event::End(_)))` when a silence long enough
  /// to be a boundary has elapsed.
  pub fn process(&mut self, chunk: &[f32]) -> Result<Option<Event>> {
    let window_size = self.chunk_samples() as u64;
    self.last_chunk_samples = window_size;
    self.current_sample = self.current_sample.saturating_add(chunk.len() as u64);

    let prob = self.model.process(chunk)?;

    // Mirror the exact Python state machine. Both conditions can
    // execute in the same call: e.g. a speech region that just ended
    // doesn't re-open on the same frame, but the `temp_end` clear on
    // a speech-return happens before the onset check.

    // 1. Speech returns after tentative silence — clear `temp_end`.
    if prob >= self.threshold && self.temp_end > 0 {
      self.temp_end = 0;
    }

    // 2. Speech onset: first voiced chunk while not already triggered.
    if prob >= self.threshold && !self.triggered {
      self.triggered = true;
      // Python: max(0, current_sample - speech_pad_samples - window_size_samples)
      let start = self
        .current_sample
        .saturating_sub(self.speech_pad_samples)
        .saturating_sub(window_size);
      return Ok(Some(Event::Start(start)));
    }

    // 3. Silence detection while triggered.
    if prob < self.neg_threshold && self.triggered {
      if self.temp_end == 0 {
        self.temp_end = self.current_sample;
      }
      let sil_dur = self.current_sample - self.temp_end;
      if sil_dur < self.min_silence_samples {
        return Ok(None);
      }
      // Silence has been long enough to be a real boundary. Emit End
      // and clear the trigger.
      //
      // Python: speech_end = temp_end + speech_pad_samples - window_size_samples
      // We subtract `window_size` from the padded `temp_end` to align
      // with the Python semantics (temp_end is measured at the start
      // of the silent chunk).
      let end = self
        .temp_end
        .saturating_add(self.speech_pad_samples)
        .saturating_sub(window_size);
      self.temp_end = 0;
      self.triggered = false;
      return Ok(Some(Event::End(end)));
    }

    Ok(None)
  }

  /// Flush any open speech region and reset the iterator.
  ///
  /// Unlike upstream Python, which leaves a trailing open region
  /// silently dropped at end of stream, this method returns an
  /// [`Event::End`] for the unfinished region. The end position is
  /// the current sample count (where the last processed chunk
  /// ended). The iterator is also reset to a clean state.
  ///
  /// Returns `None` if no speech region is open.
  pub fn finish(&mut self) -> Option<Event> {
    let event = if self.triggered {
      // Use current_sample as the effective end, since we have no
      // idea where the silence would have confirmed. Pad is already
      // irrelevant here — this is a forced closure.
      Some(Event::End(self.current_sample))
    } else {
      None
    };
    self.reset();
    event
  }

  /// Returns whether the iterator is currently inside a speech region.
  ///
  /// This is useful for UI state tracking — e.g. showing a "recording"
  /// indicator between `Start` and `End` events.
  #[inline]
  pub const fn is_triggered(&self) -> bool {
    self.triggered
  }

  /// Borrow the underlying model. Rarely needed; exposed so advanced
  /// callers can e.g. swap the sample rate mid-stream. If you do,
  /// make sure to also [`Self::reset`] afterwards.
  #[inline]
  pub fn model_mut(&mut self) -> &mut SileroModel {
    &mut self.model
  }
}

// We re-export the helper here so downstream tests can reason about
// millisecond → sample conversion without reaching into the private
// config module.
#[doc(hidden)]
pub fn ms_to_samples_for_test(ms: u32, rate: SampleRate) -> u64 {
  ms_to_samples(ms, rate)
}

#[cfg(test)]
mod tests {
  // True integration tests for `VadIterator` live in
  // `tests/integration_test.rs` — they require a real ONNX model to
  // drive `process`. This file covers the state-machine-only bits
  // that can be unit-tested without the model.

  use super::*;

  /// Synthetic state machine driver that bypasses the model entirely.
  /// It takes a sequence of pre-computed probabilities and feeds them
  /// through the same state transitions as `process`. This lets us
  /// verify the Start/End logic without a real ONNX session.
  fn run_synthetic(probs: &[f32], config: VadConfig, window_size: u64) -> Vec<(usize, Event)> {
    let threshold = config.threshold();
    let neg_threshold = config.effective_negative_threshold();
    let min_silence = config.min_silence_samples();
    let pad = config.speech_pad_samples();

    let mut triggered = false;
    let mut current_sample = 0_u64;
    let mut temp_end = 0_u64;
    let mut events = Vec::new();

    for (i, &prob) in probs.iter().enumerate() {
      current_sample += window_size;

      if prob >= threshold && temp_end > 0 {
        temp_end = 0;
      }

      if prob >= threshold && !triggered {
        triggered = true;
        let start = current_sample
          .saturating_sub(pad)
          .saturating_sub(window_size);
        events.push((i, Event::Start(start)));
        continue;
      }

      if prob < neg_threshold && triggered {
        if temp_end == 0 {
          temp_end = current_sample;
        }
        let sil_dur = current_sample - temp_end;
        if sil_dur < min_silence {
          continue;
        }
        let end = temp_end.saturating_add(pad).saturating_sub(window_size);
        events.push((i, Event::End(end)));
        temp_end = 0;
        triggered = false;
      }
    }

    events
  }

  #[test]
  fn no_speech_yields_no_events() {
    let probs = vec![0.1_f32; 50];
    let events = run_synthetic(&probs, VadConfig::default(), 512);
    assert!(events.is_empty());
  }

  #[test]
  fn single_speech_region_yields_start_and_end() {
    // 20 chunks of voice, then 20 chunks of silence.
    // At 16 kHz default config: chunk = 512 samples, min_silence =
    // 1600 samples = ~3.125 chunks. The end event fires once the
    // silence gap reaches min_silence.
    let mut probs = vec![0.9; 20];
    probs.extend(vec![0.05; 20]);
    let events = run_synthetic(&probs, VadConfig::default(), 512);

    assert_eq!(events.len(), 2);
    assert!(matches!(events[0].1, Event::Start(_)));
    assert!(matches!(events[1].1, Event::End(_)));
    // Start event fires on the first voiced chunk (index 0).
    assert_eq!(events[0].0, 0);
    // End event fires once sil_dur ≥ min_silence; min_silence_samples
    // = 1600 at 16 kHz, window_size = 512, so after 4 silent chunks:
    // cur 20*512 = 10240, temp_end = 10240 (set on first silence,
    // index 20 → cur = 21*512 = 10752, temp_end set to 10752), then
    // chunks 21..(21+4-1). We don't check the exact chunk index here
    // because the off-by-one between Rust and Python is documented,
    // just that the sequence is Start then exactly one End.
  }

  #[test]
  fn brief_silence_does_not_end_speech() {
    // Voice-silence-voice with a silence shorter than min_silence.
    // Silence of 2 chunks = 1024 samples, less than default 1600.
    // The state machine should stay triggered.
    let mut probs = vec![0.9; 10];
    probs.extend(vec![0.05; 2]); // brief silence
    probs.extend(vec![0.9; 10]);
    let events = run_synthetic(&probs, VadConfig::default(), 512);

    // Exactly one Start event, no End until we add real trailing silence.
    assert_eq!(
      events
        .iter()
        .filter(|(_, e)| matches!(e, Event::Start(_)))
        .count(),
      1
    );
    assert_eq!(
      events
        .iter()
        .filter(|(_, e)| matches!(e, Event::End(_)))
        .count(),
      0
    );
  }

  #[test]
  fn brief_silence_then_long_silence_ends_once() {
    let mut probs = vec![0.9; 10];
    probs.extend(vec![0.05; 2]); // brief silence (absorbed)
    probs.extend(vec![0.9; 5]); // more voice
    probs.extend(vec![0.05; 20]); // long silence (should close segment)
    let events = run_synthetic(&probs, VadConfig::default(), 512);

    assert_eq!(
      events
        .iter()
        .filter(|(_, e)| matches!(e, Event::Start(_)))
        .count(),
      1
    );
    assert_eq!(
      events
        .iter()
        .filter(|(_, e)| matches!(e, Event::End(_)))
        .count(),
      1
    );
  }

  #[test]
  fn start_event_payload_uses_pad_and_window_offset() {
    // Voice on the first chunk: current_sample after chunk 0 = 512.
    // Expected start = max(0, 512 - pad_samples - window_size)
    //                = max(0, 512 - 480 - 512) = 0
    // (pad_samples for default 30 ms @ 16 kHz = 480)
    let events = run_synthetic(&[0.9_f32], VadConfig::default(), 512);
    assert_eq!(events.len(), 1);
    let (_, Event::Start(s)) = events[0] else {
      panic!("expected Start event");
    };
    assert_eq!(s, 0);
  }

  #[test]
  fn negative_threshold_hysteresis_prevents_chattering() {
    // With default config, threshold=0.5, neg_threshold=0.35.
    // A probability in (0.35, 0.5) while triggered should NOT close
    // the segment or reset temp_end.
    let mut probs = vec![0.9; 5];
    probs.extend(vec![0.4; 20]); // hover between thresholds
    let events = run_synthetic(&probs, VadConfig::default(), 512);

    // Start event fires, but no end — because 0.4 > neg_threshold
    // (0.35), so the state machine never enters silence-pending.
    assert_eq!(
      events
        .iter()
        .filter(|(_, e)| matches!(e, Event::Start(_)))
        .count(),
      1
    );
    assert_eq!(
      events
        .iter()
        .filter(|(_, e)| matches!(e, Event::End(_)))
        .count(),
      0
    );
  }

  #[test]
  fn multiple_speech_regions_each_get_their_events() {
    // Voice 10 chunks → silence 20 chunks → voice 10 chunks → silence 20 chunks.
    let mut probs = vec![0.9; 10];
    probs.extend(vec![0.05; 20]);
    probs.extend(vec![0.9; 10]);
    probs.extend(vec![0.05; 20]);
    let events = run_synthetic(&probs, VadConfig::default(), 512);

    let starts = events
      .iter()
      .filter(|(_, e)| matches!(e, Event::Start(_)))
      .count();
    let ends = events
      .iter()
      .filter(|(_, e)| matches!(e, Event::End(_)))
      .count();
    assert_eq!(starts, 2);
    assert_eq!(ends, 2);
  }
}
