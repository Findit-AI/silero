#!/usr/bin/env bash
# silero-rs vs upstream Python silero-vad parity harness.
#
# Requires:
# - `cargo` + Rust toolchain (silero-parity-runner builds via path = "../..")
# - `uv` on PATH (https://docs.astral.sh/uv/) for the Python venv
# - `ffmpeg` on PATH (the Python runner shells out to it for audio loading)
#
# Usage:
#   ./tests/parity/run.sh <fixture-dir|wav-path>
#
# Accepts either a fixture directory (uses `clip_16k.wav` inside) or a
# direct WAV path.
#
# The canonical test set is dia's parity fixtures at
# /Users/user/Develop/findit-studio/dia/tests/parity/fixtures/, which
# we don't copy into this repo (they're large). See README for a
# pointer.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

usage() {
  echo "usage: $(basename "$0") <fixture-dir|wav-path>" >&2
  echo "" >&2
  echo "Examples:" >&2
  echo "  $(basename "$0") /path/to/dia/tests/parity/fixtures/01_dialogue" >&2
  echo "  $(basename "$0") /path/to/clip_16k.wav" >&2
  exit 64
}

if [ "$#" -ne 1 ]; then
  usage
fi

ARG="$1"
if [ -d "$ARG" ]; then
  CLIP="$ARG/clip_16k.wav"
elif [ -f "$ARG" ]; then
  CLIP="$ARG"
else
  echo "[run.sh] $ARG is neither a directory nor a WAV file" >&2
  exit 65
fi

if [ ! -f "$CLIP" ]; then
  echo "[run.sh] no clip at $CLIP" >&2
  exit 66
fi

ABS_CLIP="$(cd "$(dirname "$CLIP")" && pwd)/$(basename "$CLIP")"
FIXTURE_NAME="$(basename "$(dirname "$ABS_CLIP")")"
if [ "$FIXTURE_NAME" = "" ] || [ "$FIXTURE_NAME" = "/" ]; then
  FIXTURE_NAME="$(basename "$ABS_CLIP" .wav)"
fi

OUT_DIR="$SCRIPT_DIR/out"
mkdir -p "$OUT_DIR"
RUST_OUT="$OUT_DIR/silero_rs_${FIXTURE_NAME}.json"
PY_OUT="$OUT_DIR/silero_py_${FIXTURE_NAME}.json"
SCORE_OUT="$OUT_DIR/score_${FIXTURE_NAME}.json"

echo "[run.sh] clip:    $ABS_CLIP"
echo "[run.sh] outputs: $RUST_OUT, $PY_OUT, $SCORE_OUT"

# 1) uv venv for the Python side. Cached when unchanged.
cd "$SCRIPT_DIR/python"
if [ ! -d .venv ]; then
  echo "[run.sh] creating uv venv at $(pwd)/.venv ..."
  uv venv
fi
echo "[run.sh] syncing python deps (cached when unchanged) ..."
uv pip install -e . > /dev/null

# 2) Rust runner. Builds in release mode with the bundled silero ONNX
# model. ort 2.0.0-rc.12's default features include `download-binaries`
# + `copy-dylibs`, so the prebuilt ONNX Runtime ships next to the
# binary — no need for `ORT_DYLIB_PATH` (unlike whispery's harness
# which uses load-dynamic).
#
# Both runners now use the upstream Python silero-vad defaults
# verbatim (threshold 0.5, min_speech_duration_ms 250,
# min_silence_duration_ms 100, speech_pad_ms 30,
# min_silence_at_max_speech_ms 98). The previous `--min-silence-ms 132`
# crate-side override compensated for an off-by-one in
# `SpeechSegmenter::push_probability`'s silence counter; that bug was
# fixed in silero v0.3.0, so the override is no longer required.
# `--min-silence-ms` remains a CLI flag on the runner for advanced
# users who want to override.
cd "$ROOT"
echo "[run.sh] running silero-parity-runner ..."
cargo run \
  --release \
  --quiet \
  --manifest-path tests/parity/Cargo.toml \
  -p silero-parity-runner \
  --bin silero-parity-runner \
  -- "$ABS_CLIP" \
  --out "$RUST_OUT"

# 3) Python runner. Defaults match upstream silero-vad PyPI defaults,
# and the crate (v0.3.0+) now matches them too.
cd "$SCRIPT_DIR/python"
echo "[run.sh] running silero_vad_runner.py ..."
uv run python silero_vad_runner.py "$ABS_CLIP" --out "$PY_OUT"

# 4) Score. Captures the score's exit code and propagates it.
cd "$SCRIPT_DIR/python"
echo "[run.sh] scoring ..."
set +e
uv run python score.py "$RUST_OUT" "$PY_OUT" --out "$SCORE_OUT"
SCORE_RC=$?
set -e

exit $SCORE_RC
