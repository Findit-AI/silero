"""Run upstream Python `silero-vad` on a 16 kHz mono WAV; emit the raw
VAD segments as JSON in the same schema as the Rust
`silero-parity-runner`.

Why this is structurally simple:
- We call `silero_vad.get_speech_timestamps(audio, model, ...)` directly.
  That's the same entry point upstream documentation publishes; the
  Rust crate's `SpeechSegmenter` is a port of the same logic.
- Defaults match between the two runners (validated 2026-05 against
  silero-vad 6.2.1 source). See `tests/parity/README.md`.

Usage:
    uv run python silero_vad_runner.py <wav_path> --out <json_path>

All knobs are exposed as CLI flags so `run.sh` can pass exactly the
same parameter set to both runners.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from importlib.metadata import version as pkg_version
from pathlib import Path

import numpy as np
import torch
from silero_vad import get_speech_timestamps, load_silero_vad


def load_audio_16k_mono_f32(path: Path) -> np.ndarray:
    """Decode `path` to 16 kHz mono `np.float32`.

    Mirrors WhisperX's `load_audio` (whisperx/audio.py): shell out to
    `ffmpeg -nostdin -threads 0 -i <path> -f s16le -ac 1 -acodec
    pcm_s16le -ar 16000 -`, then `np.frombuffer(out, np.int16).astype(
    np.float32) / 32768.0`. The Rust runner uses `ffmpeg-next` to do
    exactly the same thing in-process. Doing the same conversion on
    both sides keeps the f32 buffer the model sees byte-identical, so
    `clip_sha256` matches across runners and any output divergence is
    the model / segmenter rather than the loader.

    `silero-vad`'s own `read_audio` uses `torchaudio.load` which goes
    through ffmpeg/sox under the hood — close enough that segments
    almost always agree, but the byte-identical path is what makes the
    parity hash check meaningful.
    """
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-threads",
        "0",
        "-i",
        str(path),
        "-f",
        "s16le",
        "-ac",
        "1",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, check=True)
    pcm = np.frombuffer(proc.stdout, dtype=np.int16)
    return pcm.astype(np.float32) / 32768.0


def sha256_f32_buffer(audio: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(audio.tobytes(order="C"))
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run upstream Python silero-vad on a 16 kHz mono WAV; emit segments as JSON."
    )
    parser.add_argument("wav_path", type=Path, help="16 kHz mono WAV (any container ffmpeg can decode).")
    parser.add_argument("--out", type=Path, default=None, help="Output JSON path (default: stdout).")
    # Defaults below match `silero_vad.get_speech_timestamps` exactly
    # (validated against silero-vad 6.2.1 — see README). They also
    # match the silero Rust crate's `SpeechOptions::default()`. Both
    # runners therefore default to apples-to-apples comparison out of
    # the box.
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--min-speech-ms", type=int, default=250)
    parser.add_argument("--min-silence-ms", type=int, default=100)
    parser.add_argument("--speech-pad-ms", type=int, default=30)
    parser.add_argument("--min-silence-at-max-speech-ms", type=int, default=98)
    parser.add_argument(
        "--max-speech-s",
        type=float,
        default=None,
        help="Max speech duration in seconds before force-split. Default: no limit (math.inf).",
    )
    parser.add_argument(
        "--backend",
        choices=("jit", "onnx"),
        default="onnx",
        help=(
            "Silero model backend. Defaults to `onnx` so the Python side runs "
            "the SAME ORT bytes the Rust crate runs (silero-vad's bundled "
            "`data/silero_vad.onnx` is byte-identical to "
            "`silero/models/silero_vad.onnx`). `jit` runs PyTorch JIT "
            "instead — useful for measuring runtime drift, but those "
            "numbers are NOT a fair Rust-vs-Python segmenter comparison "
            "because PyTorch and ORT can disagree at the FP level."
        ),
    )
    args = parser.parse_args()

    wav_path = args.wav_path.resolve()
    if not wav_path.is_file():
        print(f"WAV not found: {wav_path}", file=sys.stderr)
        return 2

    audio = load_audio_16k_mono_f32(wav_path)
    sample_rate = 16_000
    duration_s = float(len(audio)) / sample_rate
    clip_sha256 = sha256_f32_buffer(audio)

    print(
        f"[silero-vad-py] wav={wav_path} dur={duration_s:.2f}s sha256={clip_sha256[:16]} "
        f"threshold={args.threshold} max_speech_s={args.max_speech_s}",
        file=sys.stderr,
    )

    t0 = time.monotonic()
    # `load_silero_vad(onnx=...)` returns the VAD model from the
    # bundled snapshot the silero-vad PyPI package ships (silero-vad 6.x
    # bundles its own ONNX/JIT in the package itself rather than via
    # torch.hub). We pass `onnx=True` by default so both runners feed
    # identical bytes to ORT — same model, same backend — and any IoU
    # disagreement is the segmenter logic, not the inference runtime.
    use_onnx = args.backend == "onnx"
    model = load_silero_vad(onnx=use_onnx)
    backend_label = "silero_vad.onnx" if use_onnx else "silero_vad.jit"
    t_load = time.monotonic()

    audio_t = torch.from_numpy(audio)

    kwargs = dict(
        sampling_rate=sample_rate,
        threshold=args.threshold,
        min_speech_duration_ms=args.min_speech_ms,
        min_silence_duration_ms=args.min_silence_ms,
        speech_pad_ms=args.speech_pad_ms,
        min_silence_at_max_speech=args.min_silence_at_max_speech_ms,
    )
    if args.max_speech_s is not None:
        kwargs["max_speech_duration_s"] = args.max_speech_s
    else:
        # Match the silero-vad default explicitly: `float('inf')`. Pass
        # it through rather than relying on the library default so the
        # JSON output records exactly what was used.
        kwargs["max_speech_duration_s"] = math.inf

    timestamps = get_speech_timestamps(audio_t, model, **kwargs)
    t_vad = time.monotonic()

    print(
        f"[silero-vad-py] load={t_load - t0:.2f}s vad={t_vad - t_load:.2f}s "
        f"-> {len(timestamps)} segments",
        file=sys.stderr,
    )

    # `get_speech_timestamps` returns dicts with int sample indices
    # under `start`/`end` (since we don't pass `return_seconds=True`).
    # Emit both sample- and second-coordinates so score.py can choose.
    segments = []
    for ts in timestamps:
        start_sample = int(ts["start"])
        end_sample = int(ts["end"])
        segments.append(
            {
                "start_s": start_sample / sample_rate,
                "end_s": end_sample / sample_rate,
                "start_sample": start_sample,
                "end_sample": end_sample,
            }
        )

    payload = {
        "runner": "silero-vad-py",
        "silero_vad_version": _resolve_version(),
        "torch_version": torch.__version__,
        "backend": backend_label,
        "clip_path": str(wav_path),
        "clip_sha256": clip_sha256,
        "duration_s": duration_s,
        "params": {
            "threshold": args.threshold,
            "min_speech_duration_ms": args.min_speech_ms,
            "min_silence_duration_ms": args.min_silence_ms,
            "speech_pad_ms": args.speech_pad_ms,
            "min_silence_at_max_speech_ms": args.min_silence_at_max_speech_ms,
            "max_speech_s": args.max_speech_s,
            "sampling_rate": sample_rate,
            "window_size_samples": 512,
        },
        "segment_count": len(segments),
        "segments": segments,
    }

    serialized = json.dumps(payload, indent=2)
    if args.out is None:
        print(serialized)
    else:
        args.out.write_text(serialized + "\n")
        print(
            f"[silero-vad-py] wrote {len(segments)} segments to {args.out}",
            file=sys.stderr,
        )

    return 0


def _resolve_version() -> str | None:
    try:
        return pkg_version("silero-vad")
    except Exception:
        return None


if __name__ == "__main__":
    sys.exit(main())
