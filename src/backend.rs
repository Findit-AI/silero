use crate::options::SampleRate;

/// A voice-activity-detection model backend: the minimal per-frame
/// contract the detector needs to turn audio into speech probabilities.
///
/// A backend owns its own recurrent state and rolling context and turns
/// one exact-size frame of PCM into a single speech probability. The
/// backend-agnostic detection logic — [`SpeechSegmenter`], the
/// [`detect_speech_with`] one-shot helper — drives a backend through
/// this trait and never touches the underlying model, so the same
/// segmentation semantics work over the bundled ONNX backend
/// ([`Session`], behind the default `onnx` feature) or any other
/// implementation (e.g. a CoreML backend declaring a different frame
/// geometry).
///
/// # Geometry
///
/// [`frame_samples`](VadBackend::frame_samples) declares how many
/// samples make up one frame; the detector advances its timeline by
/// exactly that many samples per probability and chunks incoming PCM
/// accordingly. It is decoupled from
/// [`SampleRate::chunk_samples`] on purpose: the ONNX backend declares
/// `512` at 16 kHz (identical to `chunk_samples`), while a backend built
/// around a different artifact can declare any positive frame size (for
/// example `4096`) and reuse the same detector unchanged.
///
/// [`SpeechSegmenter`]: crate::SpeechSegmenter
/// [`detect_speech_with`]: crate::detect_speech_with
/// [`Session`]: crate::Session
pub trait VadBackend {
  /// The backend's own error type, bridged into [`crate::Error`].
  ///
  /// The detector converts a backend error into [`crate::Error`] via
  /// this bound. In-crate backends set this to [`crate::Error`] itself
  /// (an identity conversion); out-of-tree backends define their own
  /// error and provide `impl From<TheirError> for silero::Error`,
  /// wrapping it in the transparent [`crate::Error::Backend`] variant.
  type Error: Into<crate::Error>;

  /// The number of PCM samples in one model frame.
  ///
  /// Must be non-zero. The detector chunks input into frames of exactly
  /// this size and advances its sample timeline by this amount per
  /// emitted probability.
  fn frame_samples(&self) -> usize;

  /// The sample rate the backend expects its frames to be sampled at.
  ///
  /// Used to convert the segmenter's time-based options (durations) into
  /// sample counts and to stamp emitted [`SpeechSegment`]s.
  ///
  /// [`SpeechSegment`]: crate::SpeechSegment
  fn sample_rate(&self) -> SampleRate;

  /// Run inference on exactly one frame and return its speech
  /// probability in `[0, 1]`.
  ///
  /// `frame` must be exactly [`frame_samples`](VadBackend::frame_samples)
  /// long. The backend advances its own recurrent state and rolling
  /// context as a side effect, so successive calls form a single
  /// logical stream until [`reset`](VadBackend::reset) is called.
  ///
  /// # Errors
  ///
  /// Returns [`Self::Error`] if the underlying model fails to run or the
  /// frame does not satisfy the backend's contract.
  fn predict(&mut self, frame: &[f32]) -> Result<f32, Self::Error>;

  /// Clear the backend's recurrent state so the next
  /// [`predict`](VadBackend::predict) starts a fresh logical stream.
  fn reset(&mut self);
}
