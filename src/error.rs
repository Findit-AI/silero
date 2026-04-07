//! Error types for the `silero` crate.

use std::path::PathBuf;

/// All errors returned by this crate.
///
/// Most variants are 1:1 wrappers around lower-level errors from `ort`
/// and `ndarray`. A small set of variants ([`SileroError::InvalidInput`],
/// [`SileroError::UnsupportedSampleRate`], etc.) represent contract
/// violations in the public API — the caller passed an argument that
/// we can statically rule out.
#[derive(Debug, thiserror::Error)]
pub enum SileroError {
  /// Failed to load the ONNX model from the given path.
  ///
  /// The `path` is included so the error message is actionable.
  #[error("failed to load Silero VAD model from {path}: {source}")]
  LoadModel {
    /// Path the user asked us to open.
    path: PathBuf,
    /// Underlying ORT error.
    #[source]
    source: ort::Error,
  },

  /// Low-level ONNX Runtime error (session creation, inference, tensor
  /// extraction).
  #[error(transparent)]
  Ort(#[from] ort::Error),

  /// Tensor shape mismatch while packing model input. Should not occur
  /// unless the bundled ONNX model was replaced with an incompatible
  /// one.
  #[error(transparent)]
  Shape(#[from] ndarray::ShapeError),

  /// The caller passed an unsupported sample rate.
  ///
  /// Silero VAD is trained on 8 kHz and 16 kHz audio only. Higher
  /// sample rates that are integer multiples of 16 kHz are accepted by
  /// [`crate::speech_timestamps`] via automatic downsampling, matching
  /// the upstream Python behaviour; anything else must be resampled by
  /// the caller (see the optional `resample` feature).
  #[error(
    "unsupported sample rate: {rate} Hz (Silero VAD only supports 8 kHz, 16 kHz, \
     or integer multiples of 16 kHz)"
  )]
  UnsupportedSampleRate {
    /// The offending sample rate in hertz.
    rate: u32,
  },

  /// The internal ONNX session mutex was poisoned because a previous
  /// inference call panicked. The session is unusable after this.
  #[error("Silero VAD session is poisoned (a previous inference call panicked)")]
  Poisoned,

  /// The ONNX model output had an unexpected shape. Should never
  /// happen with the bundled model; present so we fail loudly instead
  /// of producing garbage if a future upstream model breaks compat.
  #[error("Silero VAD model produced output with unexpected shape: {0:?}")]
  UnexpectedOutputShape(Vec<usize>),

  /// A caller-side invariant was violated (e.g. a chunk with the wrong
  /// number of samples was passed to [`crate::SileroModel::process`]).
  #[error("invalid input to Silero VAD: {0}")]
  InvalidInput(&'static str),
}

/// Convenient [`Result`] alias for this crate.
pub type Result<T> = core::result::Result<T, SileroError>;
