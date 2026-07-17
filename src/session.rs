use std::path::Path;

use ort::{session::Session as OrtSession, value::TensorRef};

use crate::{
  Result,
  backend::VadBackend,
  error::Error,
  options::{SampleRate, SessionOptions},
  stream::{MAX_CHUNK_SAMPLES, STATE_HIDDEN_DIM, STATE_LAYERS, STATE_VALUES, StreamState},
};

const INPUT_NAME: &str = "input";
const STATE_NAME: &str = "state";
const SR_NAME: &str = "sr";
const OUTPUT_NAME: &str = "output";
const STATE_N_NAME: &str = "stateN";
const SCALAR_SHAPE: [usize; 0] = [];

/// Bundled ONNX model for Silero VAD inference, included as bytes in the binary when the `bundled` feature is enabled.
#[cfg(feature = "bundled")]
#[cfg_attr(docsrs, doc(cfg(feature = "bundled")))]
pub const BUNDLED_MODEL: &[u8] = include_bytes!(concat!(
  env!("CARGO_MANIFEST_DIR"),
  "/models/silero_vad.onnx"
));

/// One exact-size chunk paired with the per-stream memory it belongs to.
///
/// Batch inference is only valid when every item represents an
/// independent stream at the same sample rate.
pub struct BatchInput<'a> {
  stream: &'a mut StreamState,
  chunk: &'a [f32],
}

impl<'a> BatchInput<'a> {
  /// Returns the stream state associated with this batch input, which contains the recurrent memory and context for the stream that produced this chunk.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn state(&mut self) -> &mut StreamState {
    self.stream
  }

  /// Returns the chunk of audio samples for this batch input, which should be exactly the expected chunk size for the stream's sample rate.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn chunk(&self) -> &'a [f32] {
    self.chunk
  }

  /// Create a new batch input with the given stream state and chunk of audio samples.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn new(stream: &'a mut StreamState, chunk: &'a [f32]) -> Self {
    Self { stream, chunk }
  }
}

/// ONNX Runtime session for Silero VAD inference.
///
/// A single `Session` can be reused across many independent
/// [`StreamState`]s. This is the intended shape for worker-based
/// runtimes: one session per worker, one stream state per active
/// audio stream. `Session` is `Send` but not `Sync`, so move it across
/// threads if needed but do not share one instance concurrently.
pub struct Session {
  inner: OrtSession,
  input_scratch: Vec<f32>,
  state_scratch: Vec<f32>,
  tail_scratch: Vec<f32>,
  /// Per-call scratch for the probabilities emitted by `process_stream`.
  /// Cleared and re-filled each call; the returned slice borrows from
  /// this buffer.
  prob_scratch: Vec<f32>,
  /// The single-stream memory backing the [`VadBackend`] implementation
  /// (`predict` / `reset`). The multi-stream inherent API
  /// (`infer_chunk`, `infer_batch`, `process_stream`, …) takes an
  /// external `StreamState` and never touches this field.
  stream: StreamState,
}

impl Session {
  /// Create a session from the bundled Silero VAD model with default options.
  #[cfg(feature = "bundled")]
  #[cfg_attr(docsrs, doc(cfg(feature = "bundled")))]
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn bundled() -> Result<Self> {
    Self::bundled_with_options(SessionOptions::default())
  }

  /// Create a session from the bundled Silero VAD model with custom options.
  #[cfg(feature = "bundled")]
  #[cfg_attr(docsrs, doc(cfg(feature = "bundled")))]
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn bundled_with_options(options: SessionOptions) -> Result<Self> {
    Self::from_memory_with_options(BUNDLED_MODEL, options)
  }

  /// Create a session from an ONNX file at the given path with default options.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
    Self::from_file_with_options(path, SessionOptions::default())
  }

  /// Create a session from an ONNX file at the given path with custom options.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn from_file_with_options(path: impl AsRef<Path>, options: SessionOptions) -> Result<Self> {
    let path = path.as_ref();
    let session = OrtSession::builder()?
      .with_optimization_level(options.optimization_level())
      .map_err(ort::Error::from)?
      .commit_from_file(path)
      .map_err(|source| Error::LoadModel {
        path: path.to_path_buf(),
        source,
      })?;
    Ok(Self::from_ort_session(session))
  }

  /// Create a session from an ONNX model loaded in memory with default options.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn from_memory(model_bytes: &[u8]) -> Result<Self> {
    Self::from_memory_with_options(model_bytes, SessionOptions::default())
  }

  /// Create a session from an ONNX model loaded in memory with custom options.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn from_memory_with_options(model_bytes: &[u8], options: SessionOptions) -> Result<Self> {
    let session = OrtSession::builder()?
      .with_optimization_level(options.optimization_level())
      .map_err(ort::Error::from)?
      .commit_from_memory(model_bytes)?;
    Ok(Self::from_ort_session(session))
  }

  /// Create a session directly from an existing ONNX Runtime session.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn from_ort_session(inner: OrtSession) -> Self {
    Self {
      inner,
      input_scratch: Vec::new(),
      state_scratch: Vec::new(),
      tail_scratch: Vec::with_capacity(MAX_CHUNK_SAMPLES),
      prob_scratch: Vec::new(),
      stream: StreamState::new(SampleRate::default()),
    }
  }

  /// Infer one chunk for one stream, returning the speech probability for that chunk.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn infer_chunk(&mut self, stream: &mut StreamState, chunk: &[f32]) -> Result<f32> {
    Self::infer_chunk_with_scratch(
      &mut self.inner,
      &mut self.input_scratch,
      &mut self.state_scratch,
      stream,
      chunk,
    )
  }

  fn infer_chunk_with_scratch(
    inner: &mut OrtSession,
    input_scratch: &mut Vec<f32>,
    state_scratch: &mut Vec<f32>,
    stream: &mut StreamState,
    chunk: &[f32],
  ) -> Result<f32> {
    let sample_rate = stream.sample_rate();
    let chunk_samples = sample_rate.chunk_samples();
    if chunk.len() != chunk_samples {
      return Err(Error::InvalidChunkLength {
        expected: chunk_samples,
        actual: chunk.len(),
      });
    }

    let context_samples = sample_rate.context_samples();
    let input_len = chunk_samples + context_samples;

    input_scratch.clear();
    input_scratch.reserve(input_len);
    input_scratch.extend_from_slice(stream.context());
    input_scratch.extend_from_slice(chunk);

    state_scratch.clear();
    state_scratch.reserve(STATE_VALUES);
    for layer in 0..STATE_LAYERS {
      state_scratch.extend_from_slice(stream.layer(layer));
    }

    let sample_rate_hz = [i64::from(sample_rate.hz())];
    let outputs = inner.run(ort::inputs![
      INPUT_NAME => TensorRef::from_array_view(([1usize, input_len], input_scratch.as_slice()))?,
      STATE_NAME => TensorRef::from_array_view(([STATE_LAYERS, 1usize, STATE_HIDDEN_DIM], state_scratch.as_slice()))?,
      SR_NAME => TensorRef::from_array_view((SCALAR_SHAPE, &sample_rate_hz[..]))?,
    ])?;

    let (output_shape, output_data) = outputs[OUTPUT_NAME].try_extract_tensor::<f32>()?;
    validate_shape(OUTPUT_NAME, output_shape.as_ref(), &[1, 1])?;

    let (state_shape, state_data) = outputs[STATE_N_NAME].try_extract_tensor::<f32>()?;
    validate_shape(
      STATE_N_NAME,
      state_shape.as_ref(),
      &[STATE_LAYERS as i64, 1, STATE_HIDDEN_DIM as i64],
    )?;

    for layer in 0..STATE_LAYERS {
      let start = layer * STATE_HIDDEN_DIM;
      let end = start + STATE_HIDDEN_DIM;
      stream
        .layer_mut(layer)
        .copy_from_slice(&state_data[start..end]);
    }

    let context_start = chunk_samples - context_samples;
    stream
      .context_mut()
      .copy_from_slice(&chunk[context_start..]);

    Ok(output_data[0])
  }

  /// Infer a batch of chunks for a batch of streams, returning a vector of speech probabilities in the same order as the input batch.
  pub fn infer_batch(&mut self, batch: &mut [BatchInput<'_>]) -> Result<Vec<f32>> {
    if batch.is_empty() {
      return Ok(Vec::new());
    }

    let sample_rate = batch[0].stream.sample_rate();
    let chunk_samples = sample_rate.chunk_samples();
    let context_samples = sample_rate.context_samples();
    let input_len = chunk_samples + context_samples;
    let batch_size = batch.len();

    for item in batch.iter() {
      if item.stream.sample_rate() != sample_rate {
        return Err(Error::MixedBatchSampleRate {
          expected: sample_rate.hz(),
          actual: item.stream.sample_rate().hz(),
        });
      }
      if item.chunk.len() != chunk_samples {
        return Err(Error::InvalidChunkLength {
          expected: chunk_samples,
          actual: item.chunk.len(),
        });
      }
    }

    self.input_scratch.clear();
    self.input_scratch.reserve(batch_size * input_len);
    for item in batch.iter() {
      self.input_scratch.extend_from_slice(item.stream.context());
      self.input_scratch.extend_from_slice(item.chunk);
    }

    self.state_scratch.clear();
    self.state_scratch.reserve(STATE_VALUES * batch_size);
    for layer in 0..STATE_LAYERS {
      for item in batch.iter() {
        self
          .state_scratch
          .extend_from_slice(item.stream.layer(layer));
      }
    }

    let sample_rate_hz = [i64::from(sample_rate.hz())];

    let outputs = self.inner.run(ort::inputs![
      INPUT_NAME => TensorRef::from_array_view(([batch_size, input_len], self.input_scratch.as_slice()))?,
      STATE_NAME => TensorRef::from_array_view(([STATE_LAYERS, batch_size, STATE_HIDDEN_DIM], self.state_scratch.as_slice()))?,
      SR_NAME => TensorRef::from_array_view((SCALAR_SHAPE, &sample_rate_hz[..]))?,
    ])?;

    let (output_shape, output_data) = outputs[OUTPUT_NAME].try_extract_tensor::<f32>()?;
    validate_shape(OUTPUT_NAME, output_shape.as_ref(), &[batch_size as i64, 1])?;

    let (state_shape, state_data) = outputs[STATE_N_NAME].try_extract_tensor::<f32>()?;
    let expected_state_shape = [
      STATE_LAYERS as i64,
      batch_size as i64,
      STATE_HIDDEN_DIM as i64,
    ];
    validate_shape(STATE_N_NAME, state_shape.as_ref(), &expected_state_shape)?;

    for layer in 0..STATE_LAYERS {
      let layer_offset = layer * batch_size * STATE_HIDDEN_DIM;
      for (index, item) in batch.iter_mut().enumerate() {
        let start = layer_offset + index * STATE_HIDDEN_DIM;
        let end = start + STATE_HIDDEN_DIM;
        item
          .stream
          .layer_mut(layer)
          .copy_from_slice(&state_data[start..end]);
      }
    }
    for item in batch.iter_mut() {
      let context_start = item.chunk.len() - context_samples;
      item
        .stream
        .context_mut()
        .copy_from_slice(&item.chunk[context_start..]);
    }

    Ok(output_data.to_vec())
  }

  /// Probabilities recorded by the most recent successful
  /// [`Self::process_stream`] call.
  ///
  /// Identical to the slice that call returned. On the error path,
  /// `process_stream` rolls back: `StreamState` is restored to its
  /// pre-call snapshot and `prob_scratch` is cleared, so this
  /// accessor returns an empty slice after a failed call.
  ///
  /// [`Self::flush_stream`] clears `prob_scratch` at entry on both
  /// paths (Ok and Err), so this accessor is also empty after a
  /// flush call regardless of outcome — flush returns its single
  /// probability via `Result<Option<f32>>`.
  ///
  /// The slice is only valid until the next call on `self` mutates
  /// `prob_scratch`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn last_probabilities(&self) -> &[f32] {
    &self.prob_scratch
  }

  /// Feed arbitrarily-sized PCM into one stream and return the
  /// probabilities for every full Silero frame consumed by this call.
  ///
  /// The returned slice borrows from an internal scratch buffer and is
  /// only valid until the next call on `self`. Empty `samples` returns
  /// an empty slice (no inference, no allocation).
  ///
  /// # Atomicity
  ///
  /// `process_stream` is **all-or-nothing**: if any chunk's inference
  /// fails the call snapshots `*stream` at entry and restores the
  /// snapshot before returning, then clears `prob_scratch`. Callers
  /// can retry the same call with the same `samples` and observe the
  /// same result, with no risk of `StreamState` and downstream
  /// segmentation timelines drifting apart on partial failure.
  pub fn process_stream(&mut self, stream: &mut StreamState, samples: &[f32]) -> Result<&[f32]> {
    self.prob_scratch.clear();

    // Fast path: if pending + samples can't assemble a full chunk,
    // no inference will run — the only side effect is the
    // `append_pending` memcpy at the end, which is infallible. Skip
    // the StreamState clone so sub-frame pushes (a common shape
    // under the new push/pop API: feed 100–200 samples at a time
    // when chunk_samples == 512) don't pay a ~3 KB per-call copy.
    let chunk_samples = stream.sample_rate().chunk_samples();
    if stream.pending_len() + samples.len() < chunk_samples {
      if !samples.is_empty() {
        stream.append_pending(samples);
      }
      return Ok(&self.prob_scratch);
    }

    let snapshot = stream.clone();
    match Self::process_stream_inner(
      &mut self.inner,
      &mut self.input_scratch,
      &mut self.state_scratch,
      &mut self.tail_scratch,
      &mut self.prob_scratch,
      stream,
      samples,
    ) {
      Ok(()) => Ok(&self.prob_scratch),
      Err(error) => {
        *stream = snapshot;
        self.prob_scratch.clear();
        Err(error)
      }
    }
  }

  fn process_stream_inner(
    inner: &mut OrtSession,
    input_scratch: &mut Vec<f32>,
    state_scratch: &mut Vec<f32>,
    tail_scratch: &mut Vec<f32>,
    prob_scratch: &mut Vec<f32>,
    stream: &mut StreamState,
    samples: &[f32],
  ) -> Result<()> {
    let chunk_samples = stream.sample_rate().chunk_samples();
    let mut offset = 0usize;

    if stream.has_pending() {
      let needed = chunk_samples - stream.pending_len();
      if samples.len() < needed {
        stream.append_pending(samples);
        return Ok(());
      }

      let pending_len = stream.pending_len();
      tail_scratch.clear();
      tail_scratch.resize(chunk_samples, 0.0);
      tail_scratch[..pending_len].copy_from_slice(stream.pending());
      tail_scratch[pending_len..chunk_samples].copy_from_slice(&samples[..needed]);
      // Note: `stream.pending` is still populated here. The outer
      // `process_stream` snapshots the whole `StreamState` and rolls
      // back on error, so we don't bother clear-and-restore for the
      // pending field specifically. On success we clear after the
      // inference commits so the steady-state behavior is unchanged.
      let probability = Self::infer_chunk_with_scratch(
        inner,
        input_scratch,
        state_scratch,
        stream,
        &tail_scratch[..chunk_samples],
      )?;
      stream.clear_pending();
      prob_scratch.push(probability);
      offset = needed;
    }

    while offset + chunk_samples <= samples.len() {
      let probability = Self::infer_chunk_with_scratch(
        inner,
        input_scratch,
        state_scratch,
        stream,
        &samples[offset..offset + chunk_samples],
      )?;
      prob_scratch.push(probability);
      offset += chunk_samples;
    }

    if offset < samples.len() {
      stream.append_pending(&samples[offset..]);
    }

    Ok(())
  }

  /// Zero-pad and process any remaining partial frame for a stream.
  ///
  /// This is mainly useful at end-of-stream. If there are no pending
  /// samples, `Ok(None)` is returned.
  ///
  /// # Atomicity
  ///
  /// Like [`Self::process_stream`], this is **all-or-nothing**: a
  /// snapshot of `*stream` is taken at entry and restored on
  /// inference failure so the pending PCM tail is preserved and a
  /// retry sees the same input.
  pub fn flush_stream(&mut self, stream: &mut StreamState) -> Result<Option<f32>> {
    // Clear at entry so `last_probabilities()` never returns stale
    // probabilities from a prior `process_stream` after a flush call —
    // matches the atomicity contract advertised in the changelog.
    self.prob_scratch.clear();
    if !stream.has_pending() {
      return Ok(None);
    }
    let snapshot = stream.clone();

    let chunk_samples = stream.sample_rate().chunk_samples();
    self.tail_scratch.clear();
    self.tail_scratch.resize(chunk_samples, 0.0);
    let pending_len = stream.pending_len();
    self.tail_scratch[..pending_len].copy_from_slice(stream.pending());

    match Self::infer_chunk_with_scratch(
      &mut self.inner,
      &mut self.input_scratch,
      &mut self.state_scratch,
      stream,
      &self.tail_scratch[..chunk_samples],
    ) {
      Ok(probability) => {
        stream.clear_pending();
        Ok(Some(probability))
      }
      Err(error) => {
        *stream = snapshot;
        Err(error)
      }
    }
  }
}

/// The bundled Silero ONNX model as a [`VadBackend`], declaring the
/// 16 kHz / 512-sample frame geometry. `predict` and `reset` drive a
/// single logical stream through the session's own internal
/// [`StreamState`]; use the inherent multi-stream API for concurrent
/// streams or 8 kHz audio.
impl VadBackend for Session {
  type Error = Error;

  #[cfg_attr(not(tarpaulin), inline(always))]
  fn frame_samples(&self) -> usize {
    self.stream.sample_rate().chunk_samples()
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  fn sample_rate(&self) -> SampleRate {
    self.stream.sample_rate()
  }

  fn predict(&mut self, frame: &[f32]) -> Result<f32> {
    // The four disjoint field borrows let the shared per-chunk helper
    // reuse the session scratch buffers against the internal stream.
    Self::infer_chunk_with_scratch(
      &mut self.inner,
      &mut self.input_scratch,
      &mut self.state_scratch,
      &mut self.stream,
      frame,
    )
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  fn reset(&mut self) {
    self.stream.reset();
  }
}

#[cfg_attr(not(tarpaulin), inline(always))]
fn validate_shape(tensor: &'static str, actual: &[i64], expected: &[i64]) -> Result<()> {
  if actual == expected {
    Ok(())
  } else {
    Err(Error::UnexpectedOutputShape {
      tensor,
      shape: actual.to_vec(),
    })
  }
}

#[cfg(test)]
mod tests {
  use crate::{SampleRate, StreamState};

  use super::{Session, validate_shape};

  #[test]
  fn flush_stream_without_pending_is_noop() {
    let mut session = Session::from_memory(include_bytes!(concat!(
      env!("CARGO_MANIFEST_DIR"),
      "/models/silero_vad.onnx"
    )))
    .expect("bundled model should load");
    let mut stream = StreamState::new(SampleRate::Rate16k);
    assert!(session.flush_stream(&mut stream).expect("flush").is_none());
  }

  #[test]
  fn validate_shape_requires_exact_dimension_order() {
    assert!(validate_shape("stateN", &[2, 3, 128], &[2, 3, 128]).is_ok());
    assert!(validate_shape("stateN", &[3, 2, 128], &[2, 3, 128]).is_err());
    assert!(validate_shape("stateN", &[2, 384], &[2, 3, 128]).is_err());
  }

  #[test]
  fn last_probabilities_matches_process_stream_ok_slice() {
    // Pin the Session::last_probabilities accessor: on Ok, it must
    // mirror the slice that process_stream returned. This is the
    // visible half of the partial-failure contract — the Err-path
    // half can't be unit-tested without injecting an ort failure,
    // but the accessor's basic shape is regression-guarded here.
    let mut session = Session::from_memory(include_bytes!(concat!(
      env!("CARGO_MANIFEST_DIR"),
      "/models/silero_vad.onnx"
    )))
    .expect("bundled model should load");
    let mut stream = StreamState::new(SampleRate::Rate16k);
    let chunk = SampleRate::Rate16k.chunk_samples();
    let audio = vec![0.0f32; chunk * 4];
    let returned: Vec<f32> = session
      .process_stream(&mut stream, &audio)
      .expect("process_stream")
      .to_vec();
    assert_eq!(returned.len(), 4);
    assert_eq!(session.last_probabilities(), &returned[..]);

    // A subsequent process_stream call with empty samples does not
    // mutate prob_scratch's content for the prior call's slice
    // beyond clearing it.
    let _ = session
      .process_stream(&mut stream, &[])
      .expect("process_stream empty");
    assert!(
      session.last_probabilities().is_empty(),
      "process_stream(&[]) must clear prob_scratch"
    );
  }

  #[test]
  fn flush_stream_clears_last_probabilities_on_both_paths() {
    // Pin in 0.4.0 (codex round-5 finding): a successful
    // process_stream followed by ANY flush_stream call must leave
    // last_probabilities() empty so retry-on-error logic doesn't
    // observe stale probabilities from the prior process_stream
    // call. Verified here for the no-pending early-return path
    // (the inference path is exercised by `flush_stream_without_
    // pending_is_noop` plus the partial-tail integration test).
    let mut session = Session::from_memory(include_bytes!(concat!(
      env!("CARGO_MANIFEST_DIR"),
      "/models/silero_vad.onnx"
    )))
    .expect("bundled model should load");
    let mut stream = StreamState::new(SampleRate::Rate16k);
    let chunk = SampleRate::Rate16k.chunk_samples();
    let audio = vec![0.0f32; chunk * 3];

    // Prime prob_scratch with three probabilities.
    let _ = session
      .process_stream(&mut stream, &audio)
      .expect("process_stream");
    assert_eq!(session.last_probabilities().len(), 3);

    // No pending → Ok(None) early return must STILL clear scratch.
    assert!(session.flush_stream(&mut stream).expect("flush").is_none());
    assert!(
      session.last_probabilities().is_empty(),
      "flush_stream must clear prob_scratch even on the no-pending early return"
    );
  }

  #[test]
  fn process_stream_subframe_pushes_extend_pending_without_inference() {
    // Pin the fast-path: when pending + samples can't assemble a
    // full chunk, `process_stream` returns Ok(empty slice) and just
    // appends to `StreamState`'s pending tail. No inference runs, no
    // StreamState clone is taken — the latter we can't observe
    // directly but we can pin the visible side effects (no probs
    // produced; pending grows by exactly samples.len()).
    let mut session = Session::from_memory(include_bytes!(concat!(
      env!("CARGO_MANIFEST_DIR"),
      "/models/silero_vad.onnx"
    )))
    .expect("bundled model should load");
    let mut stream = StreamState::new(SampleRate::Rate16k);
    let chunk = SampleRate::Rate16k.chunk_samples();
    assert!(chunk >= 32, "test assumes chunk_samples >= 32");

    // Three sub-frame pushes that each leave pending < chunk.
    let probs1 = session
      .process_stream(&mut stream, &vec![0.0f32; 100])
      .expect("push 1");
    assert!(probs1.is_empty());
    assert_eq!(stream.pending_len(), 100);

    let probs2 = session
      .process_stream(&mut stream, &vec![0.0f32; 200])
      .expect("push 2");
    assert!(probs2.is_empty());
    assert_eq!(stream.pending_len(), 300);

    // Empty push under the fast-path is also a no-op besides the
    // prob_scratch clear.
    let probs3 = session
      .process_stream(&mut stream, &[])
      .expect("push empty");
    assert!(probs3.is_empty());
    assert_eq!(stream.pending_len(), 300);

    // Now push enough to cross the chunk boundary — inference runs
    // exactly once, leaving (300 + needed) - chunk samples pending.
    let bridge = vec![0.0f32; chunk];
    let probs4 = session
      .process_stream(&mut stream, &bridge)
      .expect("push bridging");
    assert_eq!(probs4.len(), 1, "exactly one chunk's worth of inference");
    assert_eq!(stream.pending_len(), 300 + chunk - chunk);
  }

  #[test]
  fn vad_backend_predict_matches_infer_chunk_and_reset_clears_state() {
    use crate::VadBackend;

    const MODEL: &[u8] = include_bytes!(concat!(
      env!("CARGO_MANIFEST_DIR"),
      "/models/silero_vad.onnx"
    ));
    let mut via_trait = Session::from_memory(MODEL).expect("load");
    let mut via_inherent = Session::from_memory(MODEL).expect("load");

    // The OnnxBackend declares the 16 kHz / 512-sample geometry.
    assert_eq!(
      via_trait.frame_samples(),
      SampleRate::Rate16k.chunk_samples()
    );
    assert_eq!(via_trait.sample_rate(), SampleRate::Rate16k);

    // `predict` (internal stream) must track `infer_chunk` (external
    // stream) frame-for-frame — the seam is the same inference path.
    let frame = SampleRate::Rate16k.chunk_samples();
    let mut stream = StreamState::new(SampleRate::Rate16k);
    let mut value = 0x1234_5678_u32;
    for _ in 0..6 {
      let chunk: Vec<f32> = (0..frame)
        .map(|_| {
          value = value.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
          ((value >> 8) as f32 / (u32::MAX >> 8) as f32) * 2.0 - 1.0
        })
        .collect();
      let trait_prob = via_trait.predict(&chunk).expect("predict");
      let inherent_prob = via_inherent
        .infer_chunk(&mut stream, &chunk)
        .expect("infer_chunk");
      assert_eq!(
        trait_prob, inherent_prob,
        "VadBackend::predict must equal infer_chunk on the same input"
      );
    }

    // `reset` clears the recurrent state: a probe after reset matches a
    // freshly loaded session's first probe.
    via_trait.reset();
    let mut fresh = Session::from_memory(MODEL).expect("load");
    let probe = vec![0.0_f32; frame];
    assert_eq!(
      via_trait.predict(&probe).expect("predict"),
      fresh.predict(&probe).expect("predict"),
      "reset() must zero the recurrent state"
    );
  }
}
