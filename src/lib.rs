#![doc = include_str!("../README.md")]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![cfg_attr(docsrs, allow(unused_attributes))]
#![deny(missing_docs)]
#![forbid(unsafe_code)]

mod detector;
mod error;
mod options;
mod session;
mod stream;

pub use detector::{SpeechDetector, SpeechSegment, SpeechSegmenter, detect_speech};
pub use error::{Error, Result};
pub use options::{GraphOptimizationLevel, SampleRate, SessionOptions, SpeechOptions};
pub use session::{BatchInput, Session};
pub use stream::StreamState;
