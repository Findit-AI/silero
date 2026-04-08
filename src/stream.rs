use std::ops::Range;

use crate::options::SampleRate;

pub(crate) const STATE_LAYERS: usize = 2;
pub(crate) const STATE_HIDDEN_DIM: usize = 128;
pub(crate) const STATE_VALUES: usize = STATE_LAYERS * STATE_HIDDEN_DIM;
pub(crate) const MAX_CONTEXT_SAMPLES: usize = SampleRate::Rate16k.context_samples();
pub(crate) const MAX_CHUNK_SAMPLES: usize = SampleRate::Rate16k.chunk_samples();

/// Per-stream model memory for Silero VAD.
///
/// Each independent audio stream needs its own `StreamState`, even if
/// multiple streams share the same [`crate::Session`]. This struct
/// stores only the recurrent state and chunking leftovers; it does not
/// own an ONNX session.
#[derive(Debug, Clone)]
pub struct StreamState {
  sample_rate: SampleRate,
  rnn_state: [f32; STATE_VALUES],
  context: [f32; MAX_CONTEXT_SAMPLES],
  pending: [f32; MAX_CHUNK_SAMPLES],
  pending_len: usize,
}

impl StreamState {
  /// Create a new stream state with the given sample rate.
  pub fn new(sample_rate: SampleRate) -> Self {
    Self {
      sample_rate,
      rnn_state: [0.0; STATE_VALUES],
      context: [0.0; MAX_CONTEXT_SAMPLES],
      pending: [0.0; MAX_CHUNK_SAMPLES],
      pending_len: 0,
    }
  }

  /// Returns the current sample rate, which determines the expected chunk size and context size.
  #[inline]
  pub const fn sample_rate(&self) -> SampleRate {
    self.sample_rate
  }

  /// Change the sample rate for this stream, resetting all state and pending samples.
  #[inline]
  pub fn set_sample_rate(&mut self, sample_rate: SampleRate) {
    if self.sample_rate != sample_rate {
      self.sample_rate = sample_rate;
      self.reset();
    }
  }

  /// Reset the stream state, clearing all RNN state, context, and pending samples.
  #[inline]
  pub fn reset(&mut self) {
    self.rnn_state.fill(0.0);
    self.context.fill(0.0);
    self.pending_len = 0;
  }

  /// Returns the number of pending samples that have not yet been processed into a full chunk.
  #[inline]
  pub fn pending_len(&self) -> usize {
    self.pending_len
  }

  /// Returns `true` if there are pending samples that have not yet been processed into a full chunk.
  #[inline]
  pub fn has_pending(&self) -> bool {
    self.pending_len != 0
  }

  #[inline]
  pub(crate) fn context(&self) -> &[f32] {
    &self.context[..self.sample_rate.context_samples()]
  }

  #[inline]
  pub(crate) fn context_mut(&mut self) -> &mut [f32] {
    let context_len = self.sample_rate.context_samples();
    &mut self.context[..context_len]
  }

  #[inline]
  pub(crate) fn pending(&self) -> &[f32] {
    &self.pending[..self.pending_len]
  }

  #[inline]
  pub(crate) fn append_pending(&mut self, samples: &[f32]) {
    let end = self.pending_len + samples.len();
    debug_assert!(end <= self.sample_rate.chunk_samples());
    self.pending[self.pending_len..end].copy_from_slice(samples);
    self.pending_len = end;
  }

  #[inline]
  pub(crate) fn clear_pending(&mut self) {
    self.pending_len = 0;
  }

  #[inline]
  pub(crate) fn layer(&self, layer: usize) -> &[f32] {
    &self.rnn_state[layer_range(layer)]
  }

  #[inline]
  pub(crate) fn layer_mut(&mut self, layer: usize) -> &mut [f32] {
    &mut self.rnn_state[layer_range(layer)]
  }
}

impl Default for StreamState {
  #[inline]
  fn default() -> Self {
    Self::new(SampleRate::default())
  }
}

#[inline]
fn layer_range(layer: usize) -> Range<usize> {
  let start = layer * STATE_HIDDEN_DIM;
  start..start + STATE_HIDDEN_DIM
}

#[cfg(test)]
mod tests {
  use crate::options::SampleRate;

  use super::StreamState;

  #[test]
  fn reset_clears_state_and_pending() {
    let mut state = StreamState::new(SampleRate::Rate16k);
    state.layer_mut(0).fill(1.0);
    state.context_mut().fill(1.0);
    state.append_pending(&[0.1, 0.2]);
    state.reset();
    assert!(state.layer(0).iter().all(|value| *value == 0.0));
    assert!(state.context().iter().all(|value| *value == 0.0));
    assert!(state.pending().is_empty());
  }

  #[test]
  fn sample_rate_switch_reinitializes_context_shape() {
    let mut state = StreamState::new(SampleRate::Rate16k);
    assert_eq!(state.context().len(), 64);
    state.set_sample_rate(SampleRate::Rate8k);
    assert_eq!(state.context().len(), 32);
    assert!(state.context().iter().all(|value| *value == 0.0));
  }
}
