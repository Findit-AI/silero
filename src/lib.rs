#![cfg_attr(feature = "onnx", doc = include_str!("../README.md"))]
#![cfg_attr(
  not(feature = "onnx"),
  doc = "Logic-only build of the `silero` VAD crate: the default `onnx` \
         feature is disabled, so no ONNX/`ort` backend is compiled. The \
         backend-agnostic detection surface — `SpeechSegmenter`, \
         `SpeechOptions`, `SpeechSegment`, the `VadBackend` seam, and \
         `detect_speech_with` — is available. Enable the default `onnx` \
         feature (or `bundled`) for the bundled Silero model, `Session`, \
         `StreamState`, and the one-shot `detect_speech` helper."
)]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![cfg_attr(docsrs, allow(unused_attributes))]
#![deny(missing_docs)]
#![forbid(unsafe_code)]

mod backend;
mod detector;
mod error;
mod options;
#[cfg(feature = "onnx")]
mod session;
#[cfg(feature = "onnx")]
mod stream;

pub use backend::VadBackend;
#[cfg(feature = "onnx")]
#[cfg_attr(docsrs, doc(cfg(feature = "onnx")))]
pub use detector::detect_speech;
pub use detector::{SpeechDetector, SpeechSegment, SpeechSegmenter, detect_speech_with};
pub use error::{Error, Result};
#[cfg(feature = "onnx")]
#[cfg_attr(docsrs, doc(cfg(feature = "onnx")))]
pub use options::{GraphOptimizationLevel, SessionOptions};
pub use options::{SampleRate, SpeechOptions};

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
