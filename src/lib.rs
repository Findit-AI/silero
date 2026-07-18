#![cfg_attr(feature = "onnx", doc = include_str!("../README.md"))]
#![cfg_attr(
  not(feature = "onnx"),
  doc = "Logic-only build of the `silero` VAD crate: the default `onnx` \
         feature is disabled, so no ONNX/`ort` backend is compiled. This \
         build is a thin re-export shell over the [`zuoer`] VAD core — the \
         backend-agnostic detection surface (`SpeechSegmenter`, \
         `SpeechOptions`, `SpeechSegment`, `SampleRate`, the `VadBackend` \
         seam, and `detect_speech_with`) is re-exported from `zuoer` and \
         resolves under `silero::` as before. Enable the default `onnx` \
         feature (or `bundled`) for the bundled Silero model, `Session`, \
         `StreamState`, the `SpeechSegmenterExt` streaming driver, and the \
         one-shot `detect_speech` helper."
)]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![cfg_attr(docsrs, allow(unused_attributes))]
#![deny(missing_docs)]
#![forbid(unsafe_code)]

#[cfg(feature = "onnx")]
mod detector;
mod error;
#[cfg(feature = "onnx")]
mod options;
#[cfg(feature = "onnx")]
mod session;
#[cfg(feature = "onnx")]
mod stream;

// The backend-agnostic VAD core lives in `zuoer`; re-export it so the 0.5.0
// source-level API keeps resolving under `silero::` (`silero::VadBackend`,
// `silero::SpeechSegmenter`, `silero::detect_speech_with`, the options /
// duration / sample-rate types).
pub use zuoer::{
  SampleRate, SpeechDetector, SpeechOptions, SpeechSegment, SpeechSegmenter, VadBackend,
  detect_speech_with,
};

pub use error::{Error, Result};

#[cfg(feature = "onnx")]
#[cfg_attr(docsrs, doc(cfg(feature = "onnx")))]
pub use detector::{SpeechSegmenterExt, detect_speech};
#[cfg(feature = "onnx")]
#[cfg_attr(docsrs, doc(cfg(feature = "onnx")))]
pub use options::{GraphOptimizationLevel, SessionOptions};

/// Version string of the `silero` crate (`CARGO_PKG_VERSION`).
///
/// Exposed so out-of-tree harnesses (e.g. the parity runner) can record
/// the exact silero version under test rather than the harness binary's
/// own version.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
#[cfg(feature = "bundled")]
#[cfg_attr(docsrs, doc(cfg(feature = "bundled")))]
pub use session::BUNDLED_MODEL;
#[cfg(feature = "onnx")]
#[cfg_attr(docsrs, doc(cfg(feature = "onnx")))]
pub use session::{BatchInput, Session};
#[cfg(feature = "onnx")]
#[cfg_attr(docsrs, doc(cfg(feature = "onnx")))]
pub use stream::StreamState;
