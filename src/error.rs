#[cfg(feature = "onnx")]
use std::path::PathBuf;

/// Errors that can occur during Silero VAD operations.
///
/// Marked `#[non_exhaustive]` because the set of variants depends on
/// enabled features (the ORT-typed variants require the default `onnx`
/// feature); downstream `match`es must include a `_` arm.
///
/// The backend-agnostic VAD errors — unsupported / mismatched sample rate,
/// invalid chunk length, unexpected model output shape, and the
/// [`VadBackend`](crate::VadBackend) error bridge — live in the [`zuoer`]
/// core and reach this type through the transparent [`Error::Core`]
/// variant.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
  /// A backend-agnostic VAD error from the [`zuoer`] core.
  ///
  /// Bridges every [`zuoer::Error`] variant — the sample-rate,
  /// chunk-length, output-shape, and [`zuoer::Error::Backend`] errors —
  /// into `silero::Error`. Transparent: its [`Display`](std::fmt::Display)
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
}

/// A convenient alias for results returned by Silero VAD operations, using the custom `Error` type defined above.
pub type Result<T> = std::result::Result<T, Error>;

/// Bridge a `silero::Error` into the [`zuoer`] backend-error channel.
///
/// [`zuoer::VadBackend`](crate::VadBackend) requires its associated
/// `Error: Into<zuoer::Error>`. The bundled [`Session`](crate::Session)
/// implements `VadBackend` with this crate's [`Error`] as its associated
/// error, so an ONNX inference failure surfaced through the generic
/// [`detect_speech_with`](crate::detect_speech_with) is wrapped in the
/// transparent [`zuoer::Error::Backend`] variant.
#[cfg(feature = "onnx")]
impl From<Error> for zuoer::Error {
  fn from(error: Error) -> Self {
    zuoer::Error::Backend(Box::new(error))
  }
}
