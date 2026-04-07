//! Low-level `SileroModel` — a thin, stateful wrapper around the
//! Silero VAD ONNX session.
//!
//! This is the piece of the crate that actually calls ONNX Runtime.
//! Every higher-level API ([`crate::VadIterator`],
//! [`crate::speech_timestamps`]) is built on top of it. Users that
//! want maximum control (custom state-machine, per-chunk probability
//! inspection, batched inference) can use this directly.
//!
//! # Contract mirroring upstream Python
//!
//! We mirror the exact call convention of
//! `silero_vad.utils_vad.OnnxWrapper.__call__` from
//! <https://github.com/snakers4/silero-vad>. In particular:
//!
//! 1. The ONNX model accepts `[1, context_size + chunk_size]` as its
//!    audio input, **not** just `[1, chunk_size]`. Passing a raw chunk
//!    triggers an internal "too short" branch and returns near-zero
//!    probabilities for every input. This is the single most common
//!    Silero ONNX integration mistake; if you're here debugging
//!    unexpectedly low voice probabilities, this is almost certainly
//!    why.
//!
//! 2. The 64-sample context (32 @ 8 kHz) is **rolling**: after each
//!    call, the context window is updated to hold the last
//!    `context_samples` samples of the just-submitted `[context |
//!    chunk]` input.
//!
//! 3. The LSTM state tensor `[2, 1, 128]` is **carried across calls**.
//!    Resetting it with [`SileroModel::reset`] zeroes both the state
//!    and the rolling context.
//!
//! 4. On sample-rate change, context + state are automatically reset
//!    to avoid feeding a mismatched cache to the new rate's model
//!    branch.

use std::{
  path::{Path, PathBuf},
  sync::Mutex,
};

use ndarray::{Array0, Array3, ArrayView2, ArrayView3};
use ort::{
  session::{builder::GraphOptimizationLevel, Session},
  value::TensorRef,
};

use crate::config::SampleRate;
use crate::error::{Result, SileroError};

// ── ONNX layout constants ──────────────────────────
//
// These match the exported `silero_vad.onnx` shipped with
// snakers4/silero-vad. The names are stable across v4/v5/v6 of the
// model (all produce the same ONNX op graph with these input/output
// names).

const INPUT_NAME: &str = "input";
const STATE_NAME: &str = "state";
const SR_NAME: &str = "sr";
const OUTPUT_NAME: &str = "output";
const STATE_N_NAME: &str = "stateN";

/// Size of the LSTM hidden state per layer. This is an architectural
/// constant of the Silero v5/v6 model.
const STATE_HIDDEN_DIM: usize = 128;

// ── SileroModel ────────────────────────────────────

/// A loaded Silero VAD ONNX session with the streaming state machine
/// around it.
///
/// `SileroModel` is cheap to clone conceptually (just a file path),
/// but **not** `Clone` in the Rust sense — the ONNX session is
/// expensive to create and is wrapped behind a [`Mutex`] so multiple
/// threads can share one loaded model. Create one per audio stream,
/// or share one with a mutex if throughput is unimportant.
///
/// # Typical usage
///
/// ```rust,ignore
/// use silero::{SileroModel, SampleRate};
///
/// let mut model = SileroModel::from_path("models/silero_vad.onnx")?;
/// model.set_sample_rate(SampleRate::Rate16k);
///
/// let chunk = vec![0.0_f32; SampleRate::Rate16k.chunk_samples()];
/// let prob = model.process(&chunk)?;
/// println!("voice probability: {prob:.3}");
/// # Ok::<_, silero::SileroError>(())
/// ```
pub struct SileroModel {
  session: Mutex<Session>,
  // Streaming state: LSTM hidden + rolling audio context.
  state: Array3<f32>,
  context: Vec<f32>,
  // Which sample rate the state/context was built for. `None` on a
  // freshly-created model; the first `process` call will initialise.
  current_rate: Option<SampleRate>,
  sample_rate: SampleRate,
  // Preallocated input buffer reused across chunks to avoid per-call
  // heap traffic. Capacity is `max(chunk) + max(context) = 512 + 64 =
  // 576` at 16 kHz; 288 at 8 kHz. We keep 576 to cover both.
  input_buf: Vec<f32>,
}

impl SileroModel {
  /// Load the Silero VAD ONNX model from a file path.
  ///
  /// The model defaults to 16 kHz operation; call
  /// [`Self::set_sample_rate`] to switch before the first chunk.
  ///
  /// # Errors
  ///
  /// Returns [`SileroError::LoadModel`] if the file cannot be read or
  /// is not a valid ONNX model. Returns [`SileroError::Ort`] if the
  /// session builder itself fails.
  pub fn from_path(path: impl AsRef<Path>) -> Result<Self> {
    Self::from_path_with_rate(path, SampleRate::Rate16k)
  }

  /// Load the model with a specific initial sample rate. Equivalent to
  /// [`Self::from_path`] + [`Self::set_sample_rate`] but avoids a
  /// first-call state reset.
  pub fn from_path_with_rate(path: impl AsRef<Path>, sample_rate: SampleRate) -> Result<Self> {
    let path: PathBuf = path.as_ref().to_path_buf();
    let session = Session::builder()?
      .with_optimization_level(GraphOptimizationLevel::Disable)
      .map_err(ort::Error::from)?
      .commit_from_file(&path)
      .map_err(|e| SileroError::LoadModel {
        path: path.clone(),
        source: e,
      })?;

    // Allocate state + context sized for the largest supported rate
    // (16 kHz) so switching to 8 kHz later never needs to reallocate.
    let state = Array3::<f32>::zeros((2, 1, STATE_HIDDEN_DIM));
    let context = vec![0.0_f32; sample_rate.context_samples()];
    let max_input = SampleRate::Rate16k.context_samples() + SampleRate::Rate16k.chunk_samples();
    let input_buf = vec![0.0_f32; max_input];

    Ok(Self {
      session: Mutex::new(session),
      state,
      context,
      current_rate: Some(sample_rate),
      sample_rate,
      input_buf,
    })
  }

  /// Switch the model to a different sample rate.
  ///
  /// If the new rate differs from the currently-active one, both the
  /// LSTM state and the rolling context are zeroed — this matches
  /// upstream Python, which resets both on a sample-rate change. If
  /// the rate is the same as before, this is a no-op.
  pub fn set_sample_rate(&mut self, rate: SampleRate) {
    if self.current_rate == Some(rate) {
      return;
    }
    self.sample_rate = rate;
    self.current_rate = Some(rate);
    self.reset();
  }

  /// Returns the currently-configured sample rate.
  #[inline]
  pub const fn sample_rate(&self) -> SampleRate {
    self.sample_rate
  }

  /// Returns the expected chunk size in samples for the current sample
  /// rate. Every call to [`Self::process`] must provide exactly this
  /// many samples.
  #[inline]
  pub const fn chunk_samples(&self) -> usize {
    self.sample_rate.chunk_samples()
  }

  /// Zero the LSTM state and the rolling audio context.
  ///
  /// Call this between independent audio streams so state from the
  /// previous stream does not bleed into the next. Not needed between
  /// consecutive calls on the same stream — the whole point of the
  /// state is that it carries information forward.
  pub fn reset(&mut self) {
    self.state.fill(0.0);
    self.context.clear();
    self.context.resize(self.sample_rate.context_samples(), 0.0);
  }

  /// Process one chunk of audio and return its voice probability in
  /// `[0, 1]`.
  ///
  /// `chunk.len()` must be exactly [`Self::chunk_samples`] — 512 at
  /// 16 kHz, 256 at 8 kHz. The rolling context is handled internally;
  /// callers should **not** prepend their own context.
  ///
  /// Updates the LSTM state and rolling context in place so the next
  /// call sees the correct temporal history.
  ///
  /// # Errors
  ///
  /// - [`SileroError::InvalidInput`] if `chunk.len()` is wrong.
  /// - [`SileroError::Ort`] on inference failure.
  /// - [`SileroError::Poisoned`] if a previous call panicked.
  /// - [`SileroError::UnexpectedOutputShape`] if the ONNX model
  ///   produces an output tensor this crate doesn't recognise
  ///   (should never happen with the bundled model).
  pub fn process(&mut self, chunk: &[f32]) -> Result<f32> {
    let chunk_size = self.sample_rate.chunk_samples();
    if chunk.len() != chunk_size {
      return Err(SileroError::InvalidInput(
        "chunk length does not match sample-rate-dependent chunk size \
         (expected 512 @ 16 kHz or 256 @ 8 kHz)",
      ));
    }
    let ctx_size = self.sample_rate.context_samples();
    let input_len = ctx_size + chunk_size;

    // Resize the reusable input buffer if we're smaller than needed.
    if self.input_buf.len() < input_len {
      self.input_buf.resize(input_len, 0.0);
    }
    // Compose `[context | chunk]` into `input_buf[..input_len]`. We
    // never shrink the buffer, but we always access only the first
    // `input_len` elements.
    self.input_buf[..ctx_size].copy_from_slice(&self.context);
    self.input_buf[ctx_size..input_len].copy_from_slice(chunk);

    // Build the three input tensors. The audio input is borrowed
    // directly from `self.input_buf` via `ArrayView2::from_shape`,
    // skipping the per-call `Vec<f32>` clone the previous version
    // did through `Array2::from_shape_vec`. Wall-clock performance
    // is unchanged in practice — the ~2.3 KB clone is dwarfed by
    // the ~165 µs ONNX inference cost on every chunk — but this
    // avoids one unnecessary allocation per call and removes a
    // piece of code that looked like an optimization opportunity
    // but turned out not to be one.
    let sr_tensor = Array0::<i64>::from_elem((), self.sample_rate.hz() as i64);

    let prob = {
      let mut session = self.session.lock().map_err(|_| SileroError::Poisoned)?;
      let input_view: ArrayView2<'_, f32> =
        ArrayView2::from_shape((1, input_len), &self.input_buf[..input_len])?;
      let input_ref = TensorRef::from_array_view(input_view)?;
      let state_view: ArrayView3<'_, f32> = self.state.view();
      let state_ref = TensorRef::from_array_view(state_view)?;
      let sr_ref = TensorRef::from_array_view(&sr_tensor)?;

      let outputs = session.run(ort::inputs![
        INPUT_NAME => input_ref,
        STATE_NAME => state_ref,
        SR_NAME => sr_ref,
      ])?;

      // `output` shape `[1, 1]`: voice probability for this chunk.
      let (out_shape, out_data) = outputs[OUTPUT_NAME].try_extract_tensor::<f32>()?;
      if out_data.is_empty() {
        return Err(SileroError::UnexpectedOutputShape(
          (0..out_shape.len())
            .map(|i| out_shape[i] as usize)
            .collect(),
        ));
      }
      let prob = out_data[0];

      // `stateN` shape `[2, 1, 128]`: updated LSTM state. The
      // reported shape can be variable across ORT versions, but the
      // total element count must match; we flatten and copy into our
      // owned `state` array.
      let (state_shape, state_data) = outputs[STATE_N_NAME].try_extract_tensor::<f32>()?;
      let expected = 2 * STATE_HIDDEN_DIM;
      if state_data.len() != expected {
        return Err(SileroError::UnexpectedOutputShape(
          (0..state_shape.len())
            .map(|i| state_shape[i] as usize)
            .collect(),
        ));
      }
      for (dst, src) in self.state.iter_mut().zip(state_data.iter()) {
        *dst = *src;
      }
      prob
    };

    // Update rolling context from the tail of the input we just
    // submitted. This matches the upstream Python reference, which
    // uses `self._context = x[..., -context_size:]`.
    self
      .context
      .copy_from_slice(&self.input_buf[input_len - ctx_size..input_len]);

    Ok(prob)
  }

  /// Process a full slice of audio in chunk-sized steps and collect
  /// every voice probability.
  ///
  /// Samples at the end of `audio` that do not fill a full chunk are
  /// **dropped**, matching the upstream Python convention of running
  /// the loop as `range(0, N, chunk_size)` without padding the tail
  /// (except inside `get_speech_timestamps`, which pads with zeros).
  ///
  /// This is a convenience wrapper over [`Self::process`] that keeps
  /// the calling code simple when the entire audio signal is already
  /// in memory.
  pub fn process_batch(&mut self, audio: &[f32]) -> Result<Vec<f32>> {
    let chunk_size = self.chunk_samples();
    let n_chunks = audio.len() / chunk_size;
    let mut out = Vec::with_capacity(n_chunks);
    let mut offset = 0;
    for _ in 0..n_chunks {
      out.push(self.process(&audio[offset..offset + chunk_size])?);
      offset += chunk_size;
    }
    Ok(out)
  }

  // ── Internal accessors used by iterator.rs / timestamps.rs ──

  /// Borrow the LSTM state. Only used by the higher-level modules in
  /// this crate that need to snapshot/restore state.
  #[allow(dead_code)]
  pub(crate) fn state_view(&self) -> ArrayView3<'_, f32> {
    self.state.view()
  }

  /// Borrow the context. Only used by the higher-level modules.
  #[allow(dead_code)]
  pub(crate) fn context_view(&self) -> ArrayView2<'_, f32> {
    // Reshape the flat context into a `[1, context_len]` view so it
    // matches the Python `self._context` layout exactly.
    ArrayView2::from_shape((1, self.context.len()), &self.context).unwrap()
  }
}

// ── Tests (pure helpers only; real ONNX tests live in `tests/`) ──

#[cfg(test)]
mod tests {
  // The only unit-testable behaviour in this module is constructor
  // validation, which requires a real ONNX file and therefore lives
  // in `tests/integration_test.rs`. Everything else in `SileroModel`
  // goes through the ONNX session and cannot be mocked without
  // reimplementing the session.
  //
  // We do, however, assert the ONNX layout constants here so a future
  // refactor can't silently change them.

  use super::*;

  #[test]
  fn onnx_layout_constants_are_correct() {
    assert_eq!(INPUT_NAME, "input");
    assert_eq!(STATE_NAME, "state");
    assert_eq!(SR_NAME, "sr");
    assert_eq!(OUTPUT_NAME, "output");
    assert_eq!(STATE_N_NAME, "stateN");
    assert_eq!(STATE_HIDDEN_DIM, 128);
  }
}
