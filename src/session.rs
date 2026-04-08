use std::path::Path;

use ort::{session::Session as OrtSession, value::TensorRef};

use crate::{
  Result,
  error::Error,
  options::SessionOptions,
  stream::{MAX_CHUNK_SAMPLES, STATE_HIDDEN_DIM, STATE_LAYERS, STATE_VALUES, StreamState},
};

const INPUT_NAME: &str = "input";
const STATE_NAME: &str = "state";
const SR_NAME: &str = "sr";
const OUTPUT_NAME: &str = "output";
const STATE_N_NAME: &str = "stateN";
const SCALAR_SHAPE: [usize; 0] = [];

#[cfg(feature = "bundled")]
const BUNDLED_MODEL: &[u8] = include_bytes!(concat!(
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
  #[inline]
  pub const fn state(&mut self) -> &mut StreamState {
    self.stream
  }

  /// Returns the chunk of audio samples for this batch input, which should be exactly the expected chunk size for the stream's sample rate.
  #[inline]
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
}

impl Session {
  /// Create a session from the bundled Silero VAD model with default options.
  #[cfg(feature = "bundled")]
  #[cfg_attr(docsrs, doc(cfg(feature = "bundled")))]
  pub fn bundled() -> Result<Self> {
    Self::bundled_with_options(SessionOptions::default())
  }

  /// Create a session from the bundled Silero VAD model with custom options.
  #[cfg(feature = "bundled")]
  #[cfg_attr(docsrs, doc(cfg(feature = "bundled")))]
  pub fn bundled_with_options(options: SessionOptions) -> Result<Self> {
    Self::from_memory_with_options(BUNDLED_MODEL, options)
  }

  /// Create a session from an ONNX file at the given path with default options.
  pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
    Self::from_file_with_options(path, SessionOptions::default())
  }

  /// Create a session from an ONNX file at the given path with custom options.
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
  pub fn from_memory(model_bytes: &[u8]) -> Result<Self> {
    Self::from_memory_with_options(model_bytes, SessionOptions::default())
  }

  /// Create a session from an ONNX model loaded in memory with custom options.
  pub fn from_memory_with_options(model_bytes: &[u8], options: SessionOptions) -> Result<Self> {
    let session = OrtSession::builder()?
      .with_optimization_level(options.optimization_level())
      .map_err(ort::Error::from)?
      .commit_from_memory(model_bytes)?;
    Ok(Self::from_ort_session(session))
  }

  /// Create a session directly from an existing ONNX Runtime session.
  #[inline]
  pub fn from_ort_session(inner: OrtSession) -> Self {
    Self {
      inner,
      input_scratch: Vec::new(),
      state_scratch: Vec::new(),
      tail_scratch: Vec::with_capacity(MAX_CHUNK_SAMPLES),
    }
  }

  /// Infer one chunk for one stream, returning the speech probability for that chunk.
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

  /// Feed arbitrarily-sized PCM into one stream and emit one
  /// probability per full Silero frame.
  pub fn process_stream<F>(
    &mut self,
    stream: &mut StreamState,
    samples: &[f32],
    mut on_probability: F,
  ) -> Result<usize>
  where
    F: FnMut(f32),
  {
    let chunk_samples = stream.sample_rate().chunk_samples();
    let mut offset = 0usize;
    let mut frames = 0usize;

    if stream.has_pending() {
      let needed = chunk_samples - stream.pending_len();
      if samples.len() < needed {
        stream.append_pending(samples);
        return Ok(0);
      }

      let pending_len = stream.pending_len();
      self.tail_scratch.clear();
      self.tail_scratch.resize(chunk_samples, 0.0);
      self.tail_scratch[..pending_len].copy_from_slice(stream.pending());
      self.tail_scratch[pending_len..chunk_samples].copy_from_slice(&samples[..needed]);
      stream.clear_pending();

      let probability = Self::infer_chunk_with_scratch(
        &mut self.inner,
        &mut self.input_scratch,
        &mut self.state_scratch,
        stream,
        &self.tail_scratch[..chunk_samples],
      )?;
      on_probability(probability);
      frames += 1;
      offset = needed;
    }

    while offset + chunk_samples <= samples.len() {
      let probability = self.infer_chunk(stream, &samples[offset..offset + chunk_samples])?;
      on_probability(probability);
      frames += 1;
      offset += chunk_samples;
    }

    if offset < samples.len() {
      stream.append_pending(&samples[offset..]);
    }

    Ok(frames)
  }

  /// Zero-pad and process any remaining partial frame for a stream.
  ///
  /// This is mainly useful at end-of-stream. If there are no pending
  /// samples, `Ok(None)` is returned.
  pub fn flush_stream(&mut self, stream: &mut StreamState) -> Result<Option<f32>> {
    if !stream.has_pending() {
      return Ok(None);
    }

    let chunk_samples = stream.sample_rate().chunk_samples();
    self.tail_scratch.clear();
    self.tail_scratch.resize(chunk_samples, 0.0);
    let pending_len = stream.pending_len();
    self.tail_scratch[..pending_len].copy_from_slice(stream.pending());
    stream.clear_pending();

    Self::infer_chunk_with_scratch(
      &mut self.inner,
      &mut self.input_scratch,
      &mut self.state_scratch,
      stream,
      &self.tail_scratch[..chunk_samples],
    )
    .map(Some)
  }
}

#[inline]
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
}
