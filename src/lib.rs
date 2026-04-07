//! A Rust implementation of the [Silero VAD] inference pipeline.
//!
//! [Silero VAD]: https://github.com/snakers4/silero-vad
//!
//! This crate is a faithful port of the Python reference at
//! `snakers4/silero-vad`. The neural network itself is unchanged — we
//! load the same ONNX model (`silero_vad.onnx`, ~2.3 MB) via
//! [`ort`] and run inference in Rust. Everything *around* the model
//! (preprocessing, rolling LSTM state, streaming boundary events,
//! padding, max-speech splitting, segment timestamp computation) is a
//! line-by-line port of the upstream Python reference so that the
//! same audio produces identical results in both implementations.
//!
//! # Quick start
//!
//! ```rust,ignore
//! use silero::{speech_timestamps, VadConfig};
//!
//! // 16 kHz mono f32 PCM.
//! let audio: Vec<f32> = load_audio();
//!
//! let segments = speech_timestamps(
//!     "models/silero_vad.onnx",
//!     &audio,
//!     16_000,
//!     VadConfig::default(),
//! )?;
//!
//! for segment in segments {
//!     println!(
//!         "speech {:.2}s → {:.2}s",
//!         segment.start_seconds(silero::SampleRate::Rate16k),
//!         segment.end_seconds(silero::SampleRate::Rate16k),
//!     );
//! }
//! # Ok::<_, silero::SileroError>(())
//! ```
//!
//! # Three layers of API
//!
//! The crate exposes three layers of control. Pick the lowest level
//! that still does what you need:
//!
//! 1. **[`speech_timestamps`]** — batch function, full buffer in,
//!    `Vec<SpeechSegment>` out. One-to-one with upstream Python's
//!    `get_speech_timestamps`. This is what you want for offline
//!    indexing, transcription pre-filtering, or any workflow where
//!    the whole audio is already in memory.
//!
//! 2. **[`VadIterator`]** — streaming state machine. Feed one chunk at
//!    a time, receive `Option<Event>` back. Use this for live audio
//!    input (microphone, network stream) where you want immediate
//!    speech-onset / speech-offset events rather than waiting for the
//!    whole file.
//!
//! 3. **[`SileroModel`]** — raw ONNX wrapper. Returns per-chunk voice
//!    probabilities and nothing else. Use this when you need to run
//!    your own boundary logic, inspect probabilities, or batch-process
//!    across multiple audio streams sharing one loaded model.
//!
//! # Relationship to upstream
//!
//! This crate is derived from the MIT-licensed Python reference at
//! <https://github.com/snakers4/silero-vad>.
//!
//! ```text
//! Copyright (c) 2020-present Silero Team
//! Licensed under MIT (see LICENSE-MIT and NOTICE)
//! ```
//!
//! We ship the `silero_vad.onnx` model file alongside the crate
//! (2.3 MB, MIT-licensed by Silero). The underlying model has been
//! tested against the upstream Python reference in our own
//! benchmarks — see the `benches/` directory and the
//! `audio-bench::vad_compare` tool in the sibling workspace for
//! detailed accuracy and speed measurements.
//!
//! # What this crate does *not* do
//!
//! - It does **not** re-implement the neural network from scratch.
//!   The Conv / LSTM / sigmoid operations are still executed by ONNX
//!   Runtime. A from-scratch implementation would not be faster
//!   (ORT is already hand-optimised) and would multiply the
//!   maintenance burden.
//! - It does **not** include audio I/O by default. Feed the crate
//!   decoded f32 PCM samples; if you need to read a WAV file or
//!   resample from an arbitrary rate, enable the `wav` or `resample`
//!   feature flag respectively.
//! - It does **not** detect sung vocals. Silero's training objective
//!   is conversational speech, not vocal music, and this is faithfully
//!   preserved. Pop / rap / operatic vocals are intentionally classified
//!   as non-speech. For music use cases, consider a genre classifier
//!   or CLAP-style audio-text model instead.
//!
//! # Feature flags
//!
//! - `wav` — enable [`io::read_wav`], a simple mono-f32 WAV loader
//!   backed by [`hound`]. Off by default to keep the core crate
//!   dependency-free. Enable in `Cargo.toml`:
//!
//!   ```toml
//!   [dependencies]
//!   silero = { version = "0.1", features = ["wav"] }
//!   ```
//!
//! - `resample` — enable high-quality resampling of arbitrary input
//!   sample rates via [`rubato`]. Off by default.
//!
//! - `full` — convenience alias for `wav` + `resample`.

#![cfg_attr(docsrs, feature(doc_cfg))]

// ── Public modules ─────────────────────────────────

pub mod config;
pub mod error;
pub mod iterator;
pub mod model;
pub mod timestamps;

#[cfg(feature = "wav")]
#[cfg_attr(docsrs, doc(cfg(feature = "wav")))]
pub mod io;

#[cfg(feature = "resample")]
#[cfg_attr(docsrs, doc(cfg(feature = "resample")))]
pub mod resample;

// ── Re-exports: the crate-level surface ────────────

pub use config::{SampleRate, VadConfig};
pub use error::{Result, SileroError};
pub use iterator::{Event, VadIterator};
pub use model::SileroModel;
pub use timestamps::{speech_timestamps, speech_timestamps_with_model, SpeechSegment};
