"""Compare two parity-runner JSON outputs (one silero-rs, one
silero-vad-py) and report per-segment IoU statistics.

Approach:
1. Pair segments by **sequence position** — silero VAD output is
   ordered and the two implementations run the same segmenter logic
   on the same audio, so the i-th segment from one runner corresponds
   to the i-th segment from the other when both are well-aligned. If
   the segment counts differ we still pair as far as the shorter list
   goes and count the rest as drops on the longer side.
2. For each matched pair, compute time-range IoU on `[start_s, end_s]`.
3. Emit a JSON summary on stdout (or `--out`) and a human-readable
   summary on stderr.

Why sequence-position pairing rather than text-aware Needleman-Wunsch
(the whispery harness does the latter): silero output has no text to
key on, and a near-bit-exact runner pair will produce near-identical
segment boundaries — sequence-position is the right matching when the
implementations are supposed to agree. If a Rust regression shifts the
segment count by ±1, the pairing degrades gracefully and the
`segment_count_*` fields in the summary make the divergence obvious.

Pass/fail: median IoU >= `--threshold` (default 0.95) AND the segment
counts match. The 0.95 default reflects "near-bit-equivalent" — silero
running through ORT vs PyTorch on identical inputs should produce
boundaries that round to within one frame (32 ms at 16 kHz / 512-sample
windows).

Usage:
    uv run python score.py <silero_rs.json> <silero_py.json>
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Segment:
    start_s: float
    end_s: float

    @property
    def duration_s(self) -> float:
        return max(0.0, self.end_s - self.start_s)


def _load(path: Path) -> tuple[str, list[Segment], dict]:
    payload = json.loads(path.read_text())
    segments = [
        Segment(start_s=float(s["start_s"]), end_s=float(s["end_s"]))
        for s in payload["segments"]
    ]
    return payload.get("runner", path.stem), segments, payload


def _iou(a: Segment, b: Segment) -> float:
    inter = max(0.0, min(a.end_s, b.end_s) - max(a.start_s, b.start_s))
    union = max(a.end_s, b.end_s) - min(a.start_s, b.start_s)
    if union <= 0.0:
        return 0.0
    return inter / union


def _stats(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0}
    sv = sorted(values)
    n = len(sv)
    return {
        "count": n,
        "mean": float(statistics.fmean(sv)),
        "median": float(statistics.median(sv)),
        "p10": float(sv[max(0, int(0.10 * (n - 1)))]),
        "p90": float(sv[min(n - 1, int(0.90 * (n - 1)))]),
        "min": float(sv[0]),
        "max": float(sv[-1]),
        "below_0.5": int(sum(1 for v in sv if v < 0.5)),
        "below_0.9": int(sum(1 for v in sv if v < 0.9)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Score two silero parity-runner JSON outputs against each other."
    )
    parser.add_argument("a_json", type=Path, help="First runner JSON (e.g. silero-rs).")
    parser.add_argument("b_json", type=Path, help="Second runner JSON (e.g. silero-vad-py).")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Write JSON summary here (default: stdout).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.95,
        help="Median IoU threshold for exit-code 0 (default: 0.95).",
    )
    parser.add_argument(
        "--allow-segment-count-mismatch",
        action="store_true",
        help=(
            "By default the score fails if the two runners produced different "
            "segment counts (which usually means a real boundary disagreement, "
            "not just a fractional shift). Pass this to soften the check to "
            "median-IoU only."
        ),
    )
    args = parser.parse_args()

    name_a, segs_a, payload_a = _load(args.a_json)
    name_b, segs_b, payload_b = _load(args.b_json)

    # Quick sanity: surface a clip_sha256 mismatch loudly. If the two
    # runners disagree on the input bytes, the IoU number is comparing
    # apples to oranges and any disagreement is the loader's fault, not
    # the model's.
    sha_a = payload_a.get("clip_sha256")
    sha_b = payload_b.get("clip_sha256")
    sha_match = sha_a is not None and sha_b is not None and sha_a == sha_b
    if not sha_match:
        print(
            f"[score] WARNING: clip_sha256 differs between runners: "
            f"{name_a}={sha_a[:16] if sha_a else '(none)'} vs "
            f"{name_b}={sha_b[:16] if sha_b else '(none)'} — IoU below "
            f"may reflect loader divergence rather than VAD divergence",
            file=sys.stderr,
        )

    pairs: list[tuple[int, int]] = []
    n_pairs = min(len(segs_a), len(segs_b))
    for i in range(n_pairs):
        pairs.append((i, i))
    dropped_a = max(0, len(segs_a) - n_pairs)
    dropped_b = max(0, len(segs_b) - n_pairs)

    matched = [(segs_a[i], segs_b[j], _iou(segs_a[i], segs_b[j])) for i, j in pairs]
    iou_values = [iou for _, _, iou in matched]
    iou_stats = _stats(iou_values)

    matched_sorted = sorted(matched, key=lambda t: t[2])
    worst = [
        {
            "iou": round(iou, 4),
            name_a: {
                "start_s": round(sa.start_s, 3),
                "end_s": round(sa.end_s, 3),
                "dur_s": round(sa.duration_s, 3),
            },
            name_b: {
                "start_s": round(sb.start_s, 3),
                "end_s": round(sb.end_s, 3),
                "dur_s": round(sb.duration_s, 3),
            },
        }
        for sa, sb, iou in matched_sorted[:5]
    ]

    counts_match = len(segs_a) == len(segs_b)
    median_pass = iou_stats.get("median", 0.0) >= args.threshold and len(matched) > 0
    passed = bool(median_pass and (counts_match or args.allow_segment_count_mismatch))

    summary = {
        "runner_a": name_a,
        "runner_b": name_b,
        "clip_sha256_match": sha_match,
        "segment_count_a": len(segs_a),
        "segment_count_b": len(segs_b),
        "matched_pairs": len(matched),
        "dropped_by_a": dropped_a,
        "dropped_by_b": dropped_b,
        "iou": iou_stats,
        "worst_5": worst,
        "threshold_median_iou": args.threshold,
        "counts_match": counts_match,
        "passed": passed,
    }

    serialized = json.dumps(summary, indent=2)
    if args.out is None:
        print(serialized)
    else:
        args.out.write_text(serialized + "\n")

    median = iou_stats.get("median", 0.0)
    print(
        f"\n[score] {name_a} ({len(segs_a)} segs) vs {name_b} ({len(segs_b)} segs)",
        file=sys.stderr,
    )
    print(
        f"  matched={len(matched)} dropped_a={dropped_a} dropped_b={dropped_b}",
        file=sys.stderr,
    )
    if iou_stats["count"] == 0:
        print(
            "  no matched pairs — both runners produced empty segment lists",
            file=sys.stderr,
        )
        # Empty + empty is technically a match; only fail if either side
        # had segments.
        return 0 if (len(segs_a) == 0 and len(segs_b) == 0) else 1
    print(
        f"  IoU mean={iou_stats['mean']:.4f} median={iou_stats['median']:.4f} "
        f"p10={iou_stats['p10']:.4f} p90={iou_stats['p90']:.4f} "
        f"below_0.5={iou_stats['below_0.5']} below_0.9={iou_stats['below_0.9']}",
        file=sys.stderr,
    )
    if worst:
        print("  worst 5 pairs:", file=sys.stderr)
        for w in worst:
            a = w[name_a]
            b = w[name_b]
            print(
                f"    iou={w['iou']:.3f} a=[{a['start_s']:.3f},{a['end_s']:.3f}] "
                f"({a['dur_s']:.3f}s) b=[{b['start_s']:.3f},{b['end_s']:.3f}] "
                f"({b['dur_s']:.3f}s)",
                file=sys.stderr,
            )

    pass_str = "PASS" if summary["passed"] else "FAIL"
    print(
        f"  {pass_str} (median IoU {median:.4f} vs threshold {args.threshold}, "
        f"counts {len(segs_a)} vs {len(segs_b)})",
        file=sys.stderr,
    )
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
