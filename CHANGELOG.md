# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0] - 2026-05-02

### Added

- `silero::VERSION` — public `&'static str` constant carrying the crate
  version (`env!("CARGO_PKG_VERSION")`). Exposed so out-of-tree harnesses
  can record the exact silero version they're exercising rather than
  their own binary's version.
- `tests/parity/` — manual parity harness comparing the silero crate's
  VAD output against upstream Python `silero-vad`. Ships a Rust runner
  (`silero-parity-runner`), a Python reference runner, an IoU scorer,
  and a `run.sh` driver. Not part of `cargo test`; invoked manually.
  See `tests/parity/README.md`.

### Changed

- **Behaviour change** — `SpeechSegmenter::push_probability` now closes
  speech segments when the silence counter matches the upstream Python
  `silero-vad` package's semantics. Previously the crate's silence
  counter was evaluated AFTER the current frame's contribution had been
  added to `current_sample`, while upstream Python evaluates the
  equivalent `cur_sample - temp_end` BEFORE the current frame is
  consumed. The crate's counter therefore fired one model frame
  (32 ms at 16 kHz / 512-sample windows) too early — at the default
  `min_silence_duration_ms = 100`, the crate closed a segment after 4
  consecutive low-probability frames where Python tolerates the dip and
  closes after 5. The same off-by-one applied to the
  `min_silence_at_max_speech_samples` comparator on the same code path.
  Discovered by the parity harness in `tests/parity/`.

### Migration

Callers who hand-tuned `min_silence_duration_ms` against the v0.2.x
response curve may want to subtract ~32 ms from their value to keep the
same effective behaviour against v0.3.0+. Default callers do not need
to change anything — defaults still match upstream silero-vad PyPI
defaults verbatim, and the response curve is now strictly closer to
upstream than it was in v0.2.x.

### Fixed

- *(parity harness)* `ffmpeg_init` stored its initialisation error in a
  stack-local that was only set inside the `Once::call_once` closure.
  After a failed first init, subsequent calls silently returned
  `Ok(())` because the closure no longer ran. Switched to a static
  `OnceLock<Result<(), String>>` so the init outcome is captured once
  and re-surfaced on every subsequent call.
- *(parity harness)* The Rust runner reported `silero_crate_version`
  as the parity-runner binary's own version (`0.0.0`) rather than the
  silero crate version under test. Now sourced from `silero::VERSION`.
- *(parity harness)* The Python runner emitted `params.max_speech_s`
  as `null` when `--max-speech-s` was omitted, contradicting the inline
  comment that said the JSON should record the effective value. Now
  records the effective value (`Infinity` when not overridden).

### Verified

- `cargo test`
- `cargo test --no-default-features`
- `cargo build --release`
- `tests/parity/run.sh` on the five short dia parity fixtures
  (`01_dialogue`, `02_pyannote_sample`, `03_dual_speaker`,
  `04_three_speaker`, `05_four_speaker`): median IoU 1.0000 and
  segment counts match exactly against upstream Python silero-vad
  (51/51, 4/4, 14/14, 6/6, 14/14) WITHOUT the previous
  `--min-silence-ms 132` override.

## [0.2.0] - 2026-04-21

### Added

- `serde` support for `*Options`

### Changed

- Change `u32` ms to `Duration` in `SpeechOptions`

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
