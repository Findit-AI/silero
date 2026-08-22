# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- Build against `ort 2.0.0-rc.13`. The `ort` requirement is a caret range, so a
  fresh resolution picks up `rc.13`, which marks
  `GraphOptimizationLevel` `#[non_exhaustive]`; the `serde` proxy conversion
  stopped compiling (`error[E0004]`). The conversion is now a `TryFrom` whose
  wildcard arm carries `#[allow(unreachable_patterns)]` — required because
  `rc.12` is still in range and the enum is closed there, where a bare wildcard
  is an `unreachable_patterns` warning and CI builds with `-D warnings`. The
  crate now compiles clean on both `rc.12` and `rc.13`.
- CI no longer asks for ONNX Runtime distributions that do not exist. `rc.13`
  made `ort-sys` prebuilt-distribution matching strict, so a link that requests
  an execution provider the host has no build for now fails instead of silently
  falling back to a build without it (`rc.12` fell back). `cargo hack test
  --each-feature` was requesting every EP individually plus `--all-features`,
  and the coverage job was requesting `--all-features`; both now skip the EP
  features. The EP features stay compile-checked by the `build` and `clippy`
  jobs, which never link.

### Changed

- **`serde`** — serializing a `SessionOptions` whose `optimization_level` is a
  `GraphOptimizationLevel` variant this version of `silero` does not know now
  fails with a serde error. Previously every variant was representable, so the
  question could not arise. The alternative — substituting a level `silero` can
  name — would silently rewrite a setting the caller asked for, which is not
  something a serializer should do quietly. Only reachable when `silero` is
  built against an `ort` release that has added an optimization level; the five
  levels that exist through `rc.13` round-trip unchanged.

## [0.7.0] - 2026-08-22

This is a pre-1.0 breaking release, and it is purely a dependency move:
`zuoer 0.1` -> `zuoer 0.2`. **No `.rs` file in this crate changed** — the
only edits are the `zuoer` version requirement, the crate version, and this
entry. Every break below is a `zuoer 0.2` break reaching consumers through
the types `silero` re-exports; none of them alters the default-feature
runtime behavior of `Session` / `detect_speech` / `detect_speech_with`.

### Why

`silero`'s ONNX segments are compared against other backends' segments at
the segment level, on the premise that segment assembly is a *shared*
baseline so any difference is attributable to model inference. A consumer
on `zuoer 0.2` comparing against `silero` on `zuoer 0.1` breaks that
premise by construction — the two stacks segment with different versions of
the same library, and the next real change to `zuoer`'s segmentation would
surface as inference drift that no test could distinguish from segmenter
drift. This release restores the shared baseline.

### Changed

- **Breaking** — the `zuoer` requirement moves from `0.1` to `0.2`. A
  consumer that names `zuoer` types through its own direct `zuoer`
  dependency (rather than through the re-exported `silero::zuoer`) must
  move to `0.2` in lockstep, or the two `zuoer`s will be distinct crates
  and the types will not unify.
- **Breaking** — `SpeechSegment` no longer implements `Eq`. It is now
  `zuoer::Run`, which carries `f32` mean/peak probability aggregates, and
  floats have no total equality. `PartialEq` is unchanged and now also
  compares the aggregates; `Debug`, `Clone`, and `Copy` are unchanged. The
  type never derived `Hash` or `Ord`, so no `HashSet`/`BTreeSet` usage can
  be affected — only an explicit `Eq` bound, or a `#[derive(Eq)]` on a
  struct that contains a `SpeechSegment`.
- **Breaking (serde)** — `SpeechOptions` now serializes its duration fields
  under `zuoer`'s neutral names: `min_speech_duration` ->
  `min_run_duration`, `min_silence_duration` -> `min_gap_duration`,
  `min_silence_at_max_speech` -> `min_gap_at_max_run`,
  `max_speech_duration` -> `max_run_duration`, `speech_pad` -> `pad`. Each
  neutral name carries the old speech-flavoured name as a deserialization
  alias, so profiles written by `silero 0.6` still load unchanged; only the
  serialized *output* moved. A consumer that asserts on the serialized JSON
  keys, or that hands the JSON to a non-Rust reader keyed on the old names,
  must follow. `SessionOptions` — this crate's own `serde` type — is
  untouched.
- **Breaking (behavior)** — segmentation thresholds are now floored at
  `zuoer::RunOptions::MIN_THRESHOLD` (`0.01`) instead of `0.0`, on the
  setter path and the `serde` path alike. A `start_threshold` of `0.0`
  meant "every frame opens a segment", which never closed the segment and
  derived an *inverted* hysteresis window; it is now excluded rather than
  documented. A start or end threshold in `[0.0, 0.01)`, and a non-finite
  one, are lifted to `0.01`. The crate default is `0.5` and nothing in this
  crate constructs a threshold below `0.01`, so this is unreachable for
  default callers; it is recorded because it is a silent value change for a
  caller that passes one.
- **Breaking (`Display`)** — `zuoer::Error::InvalidChunkLength`, which
  reaches `silero::Error` through the transparent `Error::Core` bridge and
  is the variant `Session::infer_chunk` / `infer_batch` raise on a
  wrong-length chunk, drops the `VAD` qualifier: `"invalid VAD chunk
  length: expected N samples, got M"` -> `"invalid chunk length: expected N
  samples, got M"`. (0.5.0 rendered `"invalid Silero chunk length: ..."`.)
  Never match on `Display` text.

### Added

- Nothing is added to `silero`'s own surface. `zuoer 0.2`'s new items reach
  consumers automatically through the existing re-exports, because
  `SpeechSegment` / `SpeechSegmenter` / `SpeechDetector` / `SpeechOptions`
  are now plain type aliases for `zuoer`'s domain-neutral `Run` /
  `RunSegmenter` / `RunOptions`:
  - `SpeechSegment::mean_probability()` / `peak_probability()` — the mean
    and peak of the frame probabilities the segment was built from,
    accumulated in O(1) per frame over the segment's raw model-frame span
    (`speech_pad` extension excluded, `min_silence_duration`-bridged frames
    included). Always finite and inside `[0, 1]` on a segmenter-emitted
    segment. This is segment confidence, and it is the natural thing for
    `detect_speech` callers to start reading.
  - `zuoer::RunOptions::MIN_THRESHOLD`, and the neutral `Run` /
    `RunSegmenter` / `RunOptions` spellings, are nameable as
    `silero::zuoer::…` without a direct `zuoer` dependency.

  Whether `silero` should re-export the neutral names under `silero::`
  alongside the `Speech*` ones is deliberately left out of this release,
  which is scoped to the version move.

### Migration

- Nothing to do for the default-feature `Session` / `StreamState` /
  `detect_speech` / `detect_speech_with` path: no signature, no default,
  and no emitted segment boundary changed.
- Depending on `zuoer` directly: move that requirement to `0.2` as well.
- Requiring `Eq` on `SpeechSegment`: drop the bound (or the `derive(Eq)` on
  the enclosing type) and rely on `PartialEq`.
- Persisting `SpeechOptions` as JSON/TOML: reading is source-compatible
  (the 0.1 names are aliases); writing now emits the neutral names, so
  update any out-of-Rust consumer of that output.
- Passing a `start_threshold` or `end_threshold` below `0.01`: pick a value
  at or above `0.01`, since anything lower is now stored as `0.01`.

### Verified

`zuoer 0.1` -> `0.2` changes no segmentation. Verified differentially
rather than by inference alone: one bundled-`Session` run per clip produces
the frame-probability sequence, and that *same* sequence is then segmented
by `zuoer 0.1` and `zuoer 0.2` side by side, so the model is held constant
and only the segmenter varies. 9 real 16 kHz clips (11 s to 16 min,
including the five `dia` parity fixtures) x 6 option sets (crate defaults,
two `max_speech_duration` force-split settings, a permissive no-pad set, a
strict set, and one just above the new `0.01` threshold floor) = 54
comparisons, 0 differing segment boundaries. The default-option segment
counts on the parity fixtures also still reproduce the counts recorded for
0.3.0 in `tests/parity/README.md` (`02_pyannote_sample` 4,
`03_dual_speaker` 14, `04_three_speaker` 6, `05_four_speaker` 14).

- `cargo fmt --all --check`
- `cargo clippy --all-features --all-targets --no-deps -- -D warnings`
- `cargo test --all-features` (11 unit + 7 integration + 3 doc)
- `cargo test --no-default-features`
- `cargo test --doc --all-features`
- `RUSTDOCFLAGS="-D warnings" cargo doc --all-features --no-deps`
- `cargo publish --dry-run --allow-dirty`

`tests/parity/` was **not** run: the harness's `ffmpeg-next 8` pin no
longer builds against the locally installed system FFmpeg (a `bindgen`
enum-variant mismatch in `ffmpeg-sys-next`). That breakage is pre-existing
and independent of this release — it fails before any `silero` code is
compiled — and the harness's `silero`-facing code is unchanged and still
type-checks against `zuoer 0.2`.

## [0.6.0] - 2026-07-19

This is a pre-1.0 breaking release. Cargo semver permits the breaks below;
they are documented here with per-break migration lines rather than papered
over. The default-feature runtime behavior of the bundled `Session` /
`detect_speech` path is unchanged.

### Changed

- **Breaking** — the backend-agnostic VAD core now lives in the
  [`zuoer`](https://github.com/findit-studio/zuoer) crate and is re-exported
  here. `VadBackend`, `SpeechSegmenter`, `SpeechDetector`, `SpeechSegment`,
  `SpeechOptions`, `SampleRate`, and `detect_speech_with` keep resolving
  under `silero::`; `cargo build --no-default-features` is now a thin
  re-export shell over `zuoer` (no `ort`). The bundled ONNX model,
  `Session`, `StreamState`, `BatchInput`, `SessionOptions`,
  `GraphOptimizationLevel`, and the `detect_speech` helper stay in this
  crate. The `zuoer` crate itself is re-exported as `silero::zuoer` so
  consumers can name its types (see the error and result breaks below)
  without adding a direct `zuoer` dependency.
- **Breaking** — `VadBackend` is now push-based. The per-frame
  `predict(&[f32]) -> Result<f32>` + `frame_samples()` contract is replaced
  by `push(&[f32], &mut dyn FnMut(f32))` + `finish(&mut dyn FnMut(f32))` +
  `frame_hop()`: a backend now owns its input windowing (overlapping
  analysis windows and delayed first output are expressible — e.g. a
  400-sample window at a 160-sample hop) and its end-of-stream
  trailing-frame policy (zero-pad the last partial frame, or drop it —
  `snip_edges`), emitting probabilities through a `sink`. The
  associated-error bound is `Error: Into<zuoer::Error>`. The bundled
  `Session` backend is behaviorally unchanged (window == hop == chunk,
  zero-padded tail). Only code that *implements* `VadBackend` for its own
  model must migrate.
- **Breaking** — `SpeechSegmenter::{push_samples, flush_stream,
  finish_stream}` moved to the new `SpeechSegmenterExt` extension trait,
  since `SpeechSegmenter` is now a foreign (`zuoer`) type. Bring it into
  scope with `use silero::SpeechSegmenterExt` to call them; behavior is
  identical (they drive the segmenter via its `push_probabilities` /
  `pop_pending` / `finish` sans-I/O seam).
- **Breaking** — `SpeechSegmenter::{frame_samples, set_frame_samples}` were
  renamed to `{frame_hop, set_frame_hop}`. Hop is the accurate concept for
  the push-based redesign: the timeline advance of a single emitted
  probability. This is the consumer-facing accessor rename on the segmenter
  itself — distinct from the `VadBackend` `frame_samples()` → `push` /
  `frame_hop()` contract change above, which only affects backend
  *implementors*. The values are unchanged for the bundled Silero geometry
  (hop == frame == chunk == 512 at 16 kHz).
- **Breaking** — `SampleRate::context_samples()` (public at 0.5.0) is gone
  from the public `SampleRate`. The Silero rolling-context geometry it
  reported (64 samples at 16 kHz, 32 at 8 kHz) is Session-specific and now
  lives crate-private on the `onnx` `Session` path; the re-exported `zuoer`
  `SampleRate` carries only backend-agnostic geometry (`hz`,
  `chunk_samples`). There is no public replacement — a caller reading
  `context_samples()` was depending on Silero-internal `Session` geometry.
- **Breaking** — four backend-agnostic `Error` variants
  (`UnsupportedSampleRate`, `IncompatibleSampleRate`, `InvalidChunkLength`,
  `Backend`) moved to `zuoer::Error` and reach `silero::Error` through the
  new transparent `Error::Core` bridge. The two Session-specific variants
  (`MixedBatchSampleRate`, `UnexpectedOutputShape`) remain in `silero::Error`
  (behind the `onnx` feature); `Error::{LoadModel, Ort}` stay. A `match` on a
  *moved* variant must go through `Error::Core(zuoer::Error::…)`; the two
  retained variants, `LoadModel`, and `Ort` still match directly on
  `silero::Error`.
- **Breaking** — the re-exported `zuoer` callables carry `zuoer`'s types, so
  `silero::SampleRate::from_hz` and `silero::detect_speech_with` now yield
  `zuoer::Error`, not `silero::Error`. A function typed `-> silero::Result<_>`
  that forwards one of these directly must wrap it (`Ok(call()?)` or
  `.map_err(silero::Error::from)`); the `From<zuoer::Error> for silero::Error`
  bridge makes both work. (`silero::Result` is still silero's own alias — only
  the *values* the re-exported `zuoer` functions produce changed crate.)
- **Breaking** — the `From<silero::Error> for zuoer::Error` bridge is now
  total and feature-free (previously `onnx`-gated), so a logic-only consumer
  built `--no-default-features` (no `ort`) can implement `VadBackend` with
  `type Error = silero::Error`.
- **Breaking (`Display`)** — the 0.5.0→0.6.0 note that claimed `Display`
  output was unchanged was wrong: two moved variants now render `zuoer`'s
  generic wording.
  - `UnsupportedSampleRate`: `"… (Silero VAD only supports 8 kHz and 16 kHz
    directly)"` → `"… (only 8 kHz and 16 kHz are supported directly)"`.
  - `InvalidChunkLength`: `"invalid Silero chunk length: …"` → `"invalid VAD
    chunk length: …"`.
  `IncompatibleSampleRate`, `MixedBatchSampleRate`, and
  `UnexpectedOutputShape` render exactly as in 0.5.0 (the latter two stayed
  silero-owned; the `IncompatibleSampleRate` string was already identical).
  Never match on `Display` text.

### Migration

- Implementing `VadBackend` yourself: replace `predict` + `frame_samples`
  with `push` + `finish` + `frame_hop`. Buffer your own partial-frame tail,
  emit one probability per completed frame via the `sink`, and choose your
  end-of-stream tail policy in `finish`. Callers of the bundled `Session`
  backend and of `detect_speech_with` need no change.
- Calling a moved streaming method (`push_samples` / `flush_stream` /
  `finish_stream`): add `use silero::SpeechSegmenterExt;`.
- Reading or overriding the segmenter's frame geometry: rename
  `SpeechSegmenter::frame_samples` → `frame_hop` and `set_frame_samples` →
  `set_frame_hop`. Same values for the bundled Silero geometry
  (hop == frame == 512), so it is a pure rename for default callers.
- Reading `SampleRate::context_samples()`: drop the call — there is no
  public replacement, it exposed Session-internal rolling-context geometry
  and reading it reached into Silero internals. The bundled `Session`
  computes this context itself, so callers of `Session` / `detect_speech` /
  `detect_speech_with` need no change.
- Matching a moved error variant: match `silero::Error::Core(inner)`, then
  `inner` against `silero::zuoer::Error::{UnsupportedSampleRate,
  IncompatibleSampleRate, InvalidChunkLength, Backend}` — no direct `zuoer`
  dependency needed, it is re-exported as `silero::zuoer`.
  `MixedBatchSampleRate` / `UnexpectedOutputShape` still match directly on
  `silero::Error`.
- A `-> silero::Result<_>` function forwarding a re-exported `zuoer`
  callable: change e.g. `SampleRate::from_hz(hz)` to
  `Ok(SampleRate::from_hz(hz)?)` (or `.map_err(silero::Error::from)`).

### Notes

- Depends on `zuoer` via a git rev pin until `zuoer` publishes `0.1.0`;
  `silero 0.6.0` will then re-pin to `zuoer = "0.1"` before publishing.

## [0.5.0] - 2026-07-18

### Added

- `VadBackend` — the backend seam. A minimal per-frame contract
  (`frame_samples`, `sample_rate`, `predict`, `reset`, and an associated
  `Error`) that the detector drives so the same segmentation semantics
  work over any VAD model, not just the bundled ONNX one.
- `detect_speech_with` — the backend-agnostic one-shot counterpart to
  `detect_speech`, chunking a full buffer into `frame_samples`-sized
  frames over any `VadBackend`.
- `Session` now implements `VadBackend` (the ONNX backend), declaring the
  16 kHz / 512-sample frame geometry and driving a single stream through
  an internal `StreamState`.
- `SpeechSegmenter::frame_samples` / `set_frame_samples` — the segmenter's
  frame geometry is decoupled from `SampleRate::chunk_samples`, so a
  backend that declares a different frame size (e.g. 4096) reuses the
  segmentation rules unchanged.
- `Error::Backend` — a transparent variant bridging any backend's
  associated error into `silero::Error`.

### Changed

- **Breaking** — `ort` is now an optional dependency behind the default
  `onnx` feature. `cargo build --no-default-features` produces a
  logic-only crate (the segmenter, options, `SampleRate`, and the
  `VadBackend` seam) with no `ort` dependency compiled at all.
- **Breaking** — the ONNX surface now requires the `onnx` feature (on by
  default via `bundled`): `Session`, `StreamState`, `BatchInput`,
  `BUNDLED_MODEL`, `SessionOptions`, `GraphOptimizationLevel`, the
  `detect_speech` helper,
  `SpeechSegmenter::{push_samples, flush_stream, finish_stream}`, and the
  `Error::{Ort, LoadModel}` variants. Default-feature callers are
  unaffected and keep the same source-level API.
- **Breaking** — `Error` is now `#[non_exhaustive]`; downstream `match`es
  must include a `_` arm.
- **Breaking** — the execution-provider passthrough features (`coreml`,
  `directml`, `cuda`, `rocm`, `tensorrt`, `openvino`) now imply `onnx`.

### Migration

Default-feature callers keep the same source-level API, with one
exception: an exhaustive `match` on `Error` must add a `_` arm, since
`Error` is now `#[non_exhaustive]` (see above). Callers who relied on
`--no-default-features` still providing `Session` (which required `ort` to
always be present) must additionally enable the `onnx` feature — or
`bundled` for the embedded model — to compile the ONNX backend, the
examples, and the integration/doc tests.

## [0.4.0] - 2026-05-09

### Changed

- **Breaking** — `rust-version` bumped from `1.85` to `1.88`. The
  new streaming API uses let-chains, which are stable since 1.88.
- **Breaking** — replaced the closure-based streaming API with a
  sans-I/O push/pop pattern that mirrors the
  [`firered-vad`](https://crates.io/crates/firered-vad) crate.
  - `Session::process_stream(stream, samples, |prob| …) -> Result<usize>`
    becomes
    `Session::process_stream(stream, samples) -> Result<&[f32]>`.
    The returned slice borrows from an internal scratch buffer and is
    valid until the next call.
  - `SpeechSegmenter::process_samples(session, stream, samples, |seg| …)`
    becomes
    `SpeechSegmenter::push_samples(session, stream, samples) -> Result<Option<SpeechSegment>>`.
    Pass `&[]` to drain segments queued by an earlier push (rare, but
    possible at force-split where one push closes more than one
    segment).
  - `SpeechSegmenter::flush_stream(session, stream, |seg| …)` and
    `finish_stream(session, stream, |seg| …)` likewise return
    `Result<Option<SpeechSegment>>`.
  - `SpeechSegmenter::finish_stream` no longer resets the segmenter so
    queued segments survive the call. Call `reset()` explicitly when
    you want to reuse the segmenter for a new stream.
  - `SpeechSegmenter::pending_segment_count()` (new) reports how many
    segments are still awaiting a `push_samples(&[])` drain.
  - `SpeechSegmenter::finish` no longer resets the segmenter — it
    enqueues the trailing segment onto `pending_segments` and pops
    the head, so undrained segments come out in order before the
    trailing one. Callers that want to start a fresh stream after
    `finish()` must call `reset()` explicitly.
  - `Session::last_probabilities()` (new) exposes the slice recorded
    by the most recent `process_stream` call. Identical to the slice
    that call returned on the `Ok` path; empty after a failed call.
- **Atomic streaming inference** — `Session::process_stream` and
  `Session::flush_stream` now snapshot `*stream` at entry and
  restore it on inference failure. On `Err`, `StreamState` is
  exactly as it was before the call, the pending PCM tail is
  preserved, and `prob_scratch` is empty. Callers can retry the
  same call with the same `samples` and observe the same result —
  no risk of `StreamState` and downstream segmentation drifting
  apart on transient ORT errors.
- `SpeechSegmenter::reset()` is no longer `const fn` — it now clears
  the new internal `pending_segments` queue, which uses `VecDeque`.

### Migration

```rust
// before (0.3.0)
segmenter.process_samples(&mut session, &mut stream, samples, |seg| {
    handle(seg);
})?;
segmenter.finish_stream(&mut session, &mut stream, |seg| handle(seg))?;
```

```rust
// after (0.4.0)
if let Some(seg) = segmenter.push_samples(&mut session, &mut stream, samples)? {
    handle(seg);
    while let Some(more) = segmenter.push_samples(&mut session, &mut stream, &[])? {
        handle(more);
    }
}
if let Some(seg) = segmenter.finish_stream(&mut session, &mut stream)? {
    handle(seg);
    while let Some(more) = segmenter.push_samples(&mut session, &mut stream, &[])? {
        handle(more);
    }
}
```

The one-shot `silero::detect_speech` helper is unchanged.

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
