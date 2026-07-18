#[cfg(feature = "onnx")]
use std::path::PathBuf;

/// Errors that can occur during Silero VAD operations.
///
/// Marked `#[non_exhaustive]` because the set of variants depends on
/// enabled features (the ORT-typed variants require the default `onnx`
/// feature) and grows as new backends bridge their errors through
/// [`Error::Backend`]; downstream `match`es must include a `_` arm.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
  /// Errors related to loading the ONNX model, including file I/O and ONNX runtime errors.
  #[cfg(feature = "onnx")]
  #[cfg_attr(docsrs, doc(cfg(feature = "onnx")))]
  #[error("failed to load Silero model from {path}: {source}")]
  LoadModel {
    /// The path that was attempted to be loaded (for context in the error message).
    path: PathBuf,
    /// The underlying error from the ONNX runtime or file I/O.
    #[source]
    source: ort::Error,
  },

  /// Errors related to invalid input data, such as mismatched sample rates or chunk sizes.
  #[cfg(feature = "onnx")]
  #[cfg_attr(docsrs, doc(cfg(feature = "onnx")))]
  #[error(transparent)]
  Ort(#[from] ort::Error),

  /// An error raised by a [`VadBackend`] implementation.
  ///
  /// The transparent bridge for the [`VadBackend::Error`] associated
  /// type: the detector wraps a backend's error here so a single
  /// [`Error`] type covers every backend. Its [`Display`] and
  /// [`source`](std::error::Error::source) delegate to the wrapped
  /// error.
  ///
  /// [`VadBackend`]: crate::VadBackend
  /// [`VadBackend::Error`]: crate::VadBackend::Error
  /// [`Display`]: std::fmt::Display
  #[error(transparent)]
  Backend(Box<dyn std::error::Error + Send + Sync + 'static>),

  /// Errors related to unsupported or incompatible sample rates.
  #[error(
    "unsupported sample rate: {rate} Hz (Silero VAD only supports 8 kHz and 16 kHz directly)"
  )]
  UnsupportedSampleRate {
    /// The unsupported sample rate in Hz.
    rate: u32,
  },

  /// Errors related to a sample-rate mismatch between the current operation/component and the provided stream or input.
  #[error(
    "stream sample rate {actual} Hz does not match expected {expected} Hz for this operation"
  )]
  IncompatibleSampleRate {
    /// The sample rate in Hz required by the current operation or component.
    expected: u32,
    /// The sample rate in Hz provided by the stream or input that caused the mismatch.
    actual: u32,
  },

  /// Errors related to batch inference containing streams with mixed sample rates.
  #[error("batch contains mixed sample rates (expected {expected} Hz, found {actual} Hz)")]
  MixedBatchSampleRate {
    /// The expected sample rate in Hz for all streams in the batch (e.g., the sample rate of the first stream).
    expected: u32,
    /// The actual sample rate in Hz from a stream that does not match the expected sample rate.
    actual: u32,
  },

  /// Errors related to invalid chunk lengths that do not match the expected chunk size for the sample rate.
  #[error("invalid Silero chunk length: expected {expected} samples, got {actual}")]
  InvalidChunkLength {
    /// The expected chunk length in samples for the given sample rate.
    expected: usize,
    /// The actual chunk length in samples that was provided.
    actual: usize,
  },

  /// Errors related to unexpected output shapes from the model during inference.
  #[error("Silero model returned unexpected shape for {tensor}: {shape:?}")]
  UnexpectedOutputShape {
    /// The name of the tensor that had an unexpected shape.
    tensor: &'static str,
    /// The actual shape of the tensor that was returned by the model.
    shape: Vec<i64>,
  },
}

/// A convenient alias for results returned by Silero VAD operations, using the custom `Error` type defined above.
pub type Result<T> = std::result::Result<T, Error>;
