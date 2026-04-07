# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.1] — 2026-04-07

Code cleanup release. No public API changes, no behavioural changes,
no measurable performance change. Released as a separate version
purely so downstream consumers can refer to a specific commit when
the change matters for their own audit trail.

### Changed

- `SileroModel::process` now passes the audio chunk into ORT via
  `ArrayView2::from_shape(&self.input_buf[..])` instead of cloning
  it through `Array2::from_shape_vec(self.input_buf[..].to_vec())`.
  This avoids one ~2.3 KB heap allocation per inference call.

  The original implementation was tagged as a future optimization
  point in the 0.1.0 review notes. After the refactor was made and
  benchmarked head-to-head, the wall-clock cost turned out to be
  identical (~165 µs per chunk on Apple Silicon, dominated by ONNX
  inference) — the clone was too small to register against the
  ORT call cost. The refactor ships anyway because:

  1. The new code is shorter and less misleading (no comment
     claiming the clone is unavoidable).
  2. One fewer allocation per call is one fewer call into the
     allocator, even if it doesn't show up in `criterion`.
  3. Removing it eliminates a piece of the codebase that looked
     like an optimization opportunity but was not, so future
     readers don't burn the same cycles investigating it.

  Verified bit-for-bit identical output on all five `audio-bench`
  vad-compare datasets after the refactor.

### Removed

- The `Array2` import in `src/model.rs` is no longer needed.

### Verified

- 26 unit tests + 9 integration tests still pass with the refactor.
- `cargo clippy --all-targets --features full -- -D warnings` is
  clean.
- `audio-bench::vad_compare::SileroVadV5` (which depends on this
  crate via path) produces bit-exact identical output on tram
  (0/0.0%), Ambience 21 (19/0.5%), 12-min dialog (144/31.2%),
  music mix (32/14.1%), and edge silence/noise (0/0.0%).

## [0.1.0] — 2026-04-07

Initial release. A complete Rust port of the Python inference pipeline
from `snakers4/silero-vad`, faithful to the upstream defaults and
behaviour.

### Added

- `SileroModel` — low-level ONNX session wrapper with rolling LSTM
  state, audio context buffer, and per-chunk inference. Mirrors
  upstream Python `OnnxWrapper`.
- `VadIterator` — streaming state machine that emits `Event::Start` /
  `Event::End` boundary events, plus a `finish()` method that flushes
  any region still open at end of stream (a deliberate addition over
  the upstream Python `VADIterator`, which silently drops them).
- `speech_timestamps` / `speech_timestamps_with_model` — batch API
  equivalent to upstream Python `get_speech_timestamps`, with the
  full state machine including `min_speech_duration_ms` /
  `min_silence_duration_ms` / `speech_pad_ms` / `max_speech_duration_ms`
  splitting / asymmetric padding rules.
- `VadConfig` — fluent builder for tuning parameters. Defaults match
  upstream Python verbatim.
- `SampleRate` enum constraining inputs to 8 kHz or 16 kHz at the
  type level. Integer multiples of 16 kHz (32, 48, 96 kHz) are
  accepted by `speech_timestamps` and decimated automatically.
- `models/silero_vad.onnx` (~2.3 MB) bundled in the crate.
- `wav` feature — minimal WAV reader via `hound` for tests and quick
  scripts.
- `resample` feature — high-quality resampler via `rubato` for
  non-integer-multiple input rates.
- `examples/detect_file.rs` — batch detection on a WAV file.
- `examples/streaming.rs` — streaming events from a WAV file.
- `examples/compare_vs_audio_bench.rs` — internal accuracy
  cross-check against the prior `audio-bench::vad_compare::SileroVadV5`
  implementation.
- 26 unit tests covering config defaults, threshold formula, padding
  edge cases, state machine transitions, and ONNX layout constants.
- 9 integration tests covering real ONNX inference: model loading,
  silence smoke test, chunk-size validation, reset determinism,
  tram (zero speech) regression, dialog speech ratio, model reuse
  across files, streaming event balance, and `finish()` flush.
- Criterion benchmarks for per-chunk model latency, batch
  `speech_timestamps`, and streaming `VadIterator`.

### Verified

- Bit-exact accuracy match with `audio-bench::vad_compare::SileroVadV5`
  on tram (0% / 0%) and Ambience (0.5% / 0.5%) datasets.
- ~32 % speech ratio on a 12-minute balanced dialog (matches the
  prior measurement of 31.2 % within tolerance).
- ~270× real-time throughput on Apple Silicon, single thread.

### Notes

- Bundled ONNX model is the official Silero v5/v6 export. License
  attribution in `NOTICE`.
- This crate uses the same `ort = "2.0.0-rc.12"` pin as our sibling
  crates so a single onnxruntime dynamic library can back the whole
  workspace.
