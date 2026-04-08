# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-04-08

### Added

- `Session` as the reusable ONNX Runtime wrapper, with support for:
  - bundled model loading behind the `bundled` feature
  - loading from file or memory
  - wrapping an externally constructed ORT session via `from_ort_session`
- `StreamState` as explicit per-stream model memory, holding recurrent state,
  rolling context, and partial-frame carry-over.
- `SpeechSegmenter` / `SpeechDetector` as a reusable probability-to-segment
  state machine with configurable thresholds, silence handling, and speech
  padding.
- `detect_speech(...)` for one-shot offline processing on a full audio buffer.
- Exact-chunk single inference and multi-stream batch inference APIs.
- `SampleRate`, `SessionOptions`, and `SpeechOptions` for the core runtime and
  segmentation contracts.
- Bundled `models/silero_vad.onnx`.
- Examples for offline file detection and streaming usage.
- Unit, integration, and doctest coverage for session loading, streaming tail
  handling, batch equivalence, silence behavior, and segmentation semantics.

### Changed

- Tightened the core streaming design around `Session`, `StreamState`, and
  `SpeechSegmenter`, replacing the earlier Python-port-oriented surface with a
  worker-friendly API that cleanly separates reusable ONNX state from
  per-stream recurrent state.
- Reduced avoidable allocations on hot paths:
  - `StreamState` now stores recurrent state, context, and pending samples in
    fixed-capacity inline buffers.
  - `Session::infer_chunk` now uses a dedicated single-stream fast path instead
    of routing through batched inference.
  - Partial-frame staging in `process_stream` / `flush_stream` now reuses a
    session-level scratch buffer.
- Hardened post-processing and model-contract validation:
  - `SpeechOptions::end_threshold()` now guarantees a valid hysteresis window
    regardless of builder call order.
  - `SpeechSegmenter::set_sample_rate()` now resets segment timeline state when
    reconfiguring a stream.
  - `stateN` output validation now checks exact tensor shape order, not only
    flattened element count.
- Examples, integration tests, and doctests now work with
  `cargo test --no-default-features` by using `Session::from_memory(...)`
  instead of assuming the `bundled` convenience constructor is available.

### Removed

- The hard dependency on `ndarray` from the public crate implementation. The
  current runtime path now feeds ORT directly from borrowed slices.

### Verified

- `cargo test`
- `cargo test --no-default-features`

### Notes

- The crate intentionally does not own queueing, worker orchestration, health
  checks, or ORT thread-count policy. Those concerns are expected to live in a
  higher-level service crate.
- Direct model support is limited to 8 kHz and 16 kHz PCM input.
