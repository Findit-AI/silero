# silero parity test harness

A side-by-side runner that compares this crate's VAD output against
upstream Python `silero-vad` on the same audio, reporting per-segment
IoU. Models the same approach `dia/tests/parity/` uses for pyannote
parity.

The bundled ONNX model in `models/silero_vad.onnx` is the same network
upstream silero-vad ships, so this is genuinely a runtime comparison
(ORT inference + Rust segmenter vs PyTorch / ORT inference + Python
segmenter on identical bytes) — not a model-architecture comparison.

## Layout

- `Cargo.toml` / `src/main.rs` — Rust binary `silero-parity-runner`
  that loads a 16 kHz mono WAV via `ffmpeg-next`, runs
  `silero::detect_speech`, and emits JSON.
- `python/pyproject.toml` / `python/silero_vad_runner.py` — same CLI
  shape, same JSON schema, runs upstream `silero_vad.get_speech_timestamps`.
- `python/score.py` — sequence-position pairing, per-segment IoU,
  median + p10/p90 + worst-N report.
- `run.sh` — end-to-end driver (bring up venv → run both → score).

## Prerequisites

- `cargo` + Rust toolchain (the runner builds via `path = "../.."`).
- `uv` for Python virtualenv management (`brew install uv` or
  `pip install uv`).
- `ffmpeg` on PATH — the Python runner shells out to it for audio
  loading; the Rust runner uses `ffmpeg-next` (in-process bindings).
  On macOS with Homebrew FFmpeg 8.x, `ffmpeg-next` is pinned to `8`
  in `Cargo.toml` because the `7.x` series still references the
  removed `libavcodec/avfft.h` header.
- A 16 kHz mono WAV (or any container ffmpeg can decode; will be
  resampled).

ORT runtime: this crate (and therefore the runner) uses `ort` with its
default `download-binaries` + `copy-dylibs` features, so a prebuilt
ONNX Runtime ships next to the binary — `ORT_DYLIB_PATH` is **not**
required (unlike the whispery harness, which uses `load-dynamic`).

## Run

```bash
cd silero
./tests/parity/run.sh /path/to/clip_16k.wav
./tests/parity/run.sh /path/to/fixture-dir         # uses clip_16k.wav inside
```

Outputs land in `tests/parity/out/`:
- `silero_rs_<name>.json` — Rust runner output.
- `silero_py_<name>.json` — Python runner output.
- `score_<name>.json` — IoU summary.

Exit code 0 iff median IoU >= 0.95 **and** segment counts match.

## Canonical fixture set

The dia parity fixtures double as the silero parity fixtures: they're
real-speech 16 kHz mono WAVs of varying length and speaker counts.

```
/Users/user/Develop/findit-studio/dia/tests/parity/fixtures/
├── 01_dialogue/clip_16k.wav        # ~120 s, 2 spk dialogue
├── 02_pyannote_sample/clip_16k.wav # ~30 s, pyannote sample
├── 03_dual_speaker/clip_16k.wav    # ~60 s, 2 spk
├── 04_three_speaker/clip_16k.wav   # 3 spk
├── 05_four_speaker/clip_16k.wav    # 4 spk
└── 06_long_recording/clip_16k.wav  # ~977 s, long-form
```

These are deliberately **not copied** into the silero repo (they're
large; dia is the source of truth for them). Pass the directory or
WAV path on the `run.sh` command line.

For first validation we recommend the five short fixtures (skip
`06_long_recording` — at ~16 minutes it's slow to run and the short
fixtures cover all interesting boundary conditions).

## Default parameter alignment

Both runners default to the same parameter set, validated 2026-05
against `silero-vad 6.2.1` source
(`src/silero_vad/utils_vad.py:get_speech_timestamps`):

| Parameter                    | silero crate default | silero-vad-py default | Aligned? |
|------------------------------|----------------------|-----------------------|----------|
| `threshold`                  | 0.5                  | 0.5                   | yes      |
| `min_speech_duration_ms`     | 250                  | 250                   | yes      |
| `min_silence_duration_ms`    | 100                  | 100                   | yes      |
| `speech_pad_ms`              | 30                   | 30                    | yes      |
| `min_silence_at_max_speech_ms`| 98                  | 98                    | yes      |
| `max_speech_duration_s`      | None (no limit)      | `float('inf')`        | yes      |
| `sampling_rate`              | 16 000 Hz            | 16 000 Hz             | yes      |
| `window_size_samples`        | 512 (chunk_samples)  | 512                   | yes      |
| `neg_threshold` (end_thresh) | start - 0.15 (clamped to >=0.01) | start - 0.15 | yes |

(See `silero/src/options.rs:default_*` constants and the upstream
`get_speech_timestamps` function signature.)

### Off-by-one silence threshold finding (fixed in v0.3.0)

> **Status: fixed in silero v0.3.0.** The harness no longer applies
> the `--min-silence-ms 132` workaround described below. Both runners
> now use upstream Python `silero-vad`'s defaults verbatim.

**Historical context (preserved here as a record of how the bug was
characterised before the fix):**

Up to and including silero v0.2.x the crate's
`SpeechSegmenter::push_probability` and Python's
`get_speech_timestamps` differed by exactly **one model frame
(32 ms at 16 kHz / 512-sample windows)** in how they computed the
"silence so far" counter:

- **Python** (`silero_vad/utils_vad.py`):
  - `temp_end` is set to `cur_sample` on the FIRST low-probability
    frame.
  - `sil_dur_now = cur_sample - temp_end` is computed BEFORE the
    current frame is "consumed" (it's the frame's *start* sample).
  - On the first low-prob frame, `sil_dur_now = 0`. On the k-th
    consecutive low-prob frame, `sil_dur_now = (k-1) * 512`.
  - Closes when `sil_dur_now >= 1600` → k = 5 frames.

- **silero crate (pre-v0.3.0)** (`silero/src/detector.rs:147-190`):
  - `tentative_end` is set to `frame_start` on the first low-prob
    frame; immediately after, `current_sample` is incremented by
    `frame_samples` (so it represents the END of the current frame).
  - `silence_samples = current_sample - silence_start = j * 512` after
    the j-th consecutive low-prob frame (j ≥ 1).
  - Closes when `silence_samples >= 1600` → j = 4 frames.

So a 4-frame (128 ms) silence dip closed the pre-v0.3.0 crate's
segment but was *tolerated* by Python — Python kept it as one segment
until 5 consecutive low-prob frames had passed. On a clip with many
short silence dips (e.g. dialogue with quick turn-taking), the crate
produced measurably more segments than Python at the same nominal
`min_silence_duration_ms`.

**Pre-v0.3.0 workaround (now removed)**: `run.sh` used to override
the crate side with `--min-silence-ms 132` (= 100 + 32), shifting the
close threshold by one frame so the two segmenters consumed the same
number of low-prob frames before closing.

**Fix in v0.3.0**: `SpeechSegmenter::push_probability` now evaluates
the silence counter against `frame_start` (the start sample of the
current frame) instead of `current_sample` (the end). This mirrors
Python's "compute `cur_sample - temp_end` before consuming the
current frame" semantics literally. The same correction applies to
the `min_silence_at_max_speech_samples` comparator that lives on the
same code path. Both close-after-5-frames and the (4-frame, no-close)
boundary are now pinned by unit tests in
`silero/src/detector.rs::tests` —
`five_frame_silence_dip_closes_segment_at_default_min_silence` and
`four_frame_silence_dip_does_not_close_segment_at_default_min_silence`.

**Migration note for callers**: this is a behaviour change. Anyone
who hand-tuned `min_silence_duration_ms` against the v0.2.x response
curve may want to subtract ~32 ms from their override to get the
same effective behaviour against v0.3.0+.

The other parameters (start/end threshold, min-speech, speech-pad,
min-silence-at-max-speech) all lined up at defaults pre-fix too —
only the silence-counter equation diverged.

## How parity is scored

`score.py` pairs segments by **sequence position** (i-th from a vs
i-th from b) and computes time-range IoU per pair. This is the right
matcher when both runners are expected to produce the same boundaries
on the same audio — a single missing or extra segment will degrade
the metric instead of accidentally re-aligning everything around the
gap.

Pass condition (default): median IoU >= 0.95 **and** `len(segments_a)
== len(segments_b)`. Pass `--allow-segment-count-mismatch` to soften
the count check (useful when diagnosing which side over- or
under-segments).

The `clip_sha256` field on each runner output hashes the f32 PCM
bytes the model actually saw. If those hashes diverge, score.py warns
loudly because any IoU disagreement could then be a loader issue
rather than a model issue.

## Notes

- The harness is **NOT** part of `cargo test`. It's a manual run for
  release-time validation and for diagnosing regressions.
- Don't commit binary fixtures or model files into this crate.
- Don't change anything in `silero/src/` from this harness — it's
  read-only on the public crate API.
