#[cfg(feature = "onnx")]
use std::path::PathBuf;

/// Errors that can occur during Silero VAD operations.
///
/// Marked `#[non_exhaustive]` because the set of variants depends on
/// enabled features (the ORT-typed variants require the default `onnx`
/// feature); downstream `match`es must include a `_` arm.
///
/// The backend-agnostic VAD errors — unsupported / mismatched sample rate,
/// invalid chunk length, and the [`VadBackend`](crate::VadBackend) error
/// bridge — live in the [`zuoer`] core and reach this type through the
/// transparent [`Error::Core`] variant. The Session-specific batch and
/// model-output-shape errors are defined here.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
  /// A backend-agnostic VAD error from the [`zuoer`] core.
  ///
  /// Bridges every [`zuoer::Error`] variant — the sample-rate,
  /// chunk-length, and [`zuoer::Error::Backend`] errors — into
  /// `silero::Error`. Transparent: its [`Display`](std::fmt::Display)
  /// and [`source`](std::error::Error::source) delegate to the wrapped
  /// `zuoer::Error`.
  #[error(transparent)]
  Core(#[from] zuoer::Error),

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

  /// Errors related to batch inference containing streams with mixed sample rates.
  ///
  /// Session-specific: constructed only by the batched ONNX inference path
  /// ([`Session::infer_batch`](crate::Session::infer_batch)). Kept in
  /// `silero::Error` rather than the backend-agnostic `zuoer` core because
  /// it describes a batching operation only the Silero session exposes.
  #[cfg(feature = "onnx")]
  #[cfg_attr(docsrs, doc(cfg(feature = "onnx")))]
  #[error("batch contains mixed sample rates (expected {expected} Hz, found {actual} Hz)")]
  MixedBatchSampleRate {
    /// The expected sample rate in Hz for all streams in the batch (e.g., the sample rate of the first stream).
    expected: u32,
    /// The actual sample rate in Hz from a stream that does not match the expected sample rate.
    actual: u32,
  },

  /// Errors related to unexpected output shapes from the model during inference.
  ///
  /// Session-specific: constructed only when the ONNX model returns a
  /// tensor whose shape does not match the contract. Kept here rather than
  /// in the `zuoer` core because it describes this model's output tensors.
  #[cfg(feature = "onnx")]
  #[cfg_attr(docsrs, doc(cfg(feature = "onnx")))]
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

/// Bridge a `silero::Error` into the [`zuoer`] backend-error channel.
///
/// [`zuoer::VadBackend`](crate::VadBackend) requires its associated
/// `Error: Into<zuoer::Error>`. This impl is **total and feature-free**: a
/// logic-only consumer (built `--no-default-features`, no `ort` compiled)
/// can still implement `VadBackend` with `type Error = silero::Error` — the
/// bundled [`Session`](crate::Session) is only one such backend. A backend
/// error surfaced through the generic
/// [`detect_speech_with`](crate::detect_speech_with) is wrapped in the
/// transparent [`zuoer::Error::Backend`] variant.
impl From<Error> for zuoer::Error {
  fn from(error: Error) -> Self {
    zuoer::Error::Backend(Box::new(error))
  }
}
