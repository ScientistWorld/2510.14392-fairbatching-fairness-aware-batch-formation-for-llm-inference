"""Visible train-slice evaluator for scheduler output artifacts.

This evaluator is method-agnostic: it reads JSON files produced under an
outputs directory, re-aggregates their per-request records on a fixed train
slice, and writes a scores JSON file. It intentionally does not import from
method/ or from eval/test/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


SLICE = "train"
SPLIT_VERSION = "request_hash_v1"


def percentile(values: list[float], pct: float) -> float:
    values = sorted(values)
    if not values:
        return 0.0
    idx = (len(values) - 1) * pct / 100.0
    lo = int(idx)
    hi = min(lo + 1, len(values) - 1)
    frac = idx - lo
    return values[lo] * (1 - frac) + values[hi] * frac


def split_bucket(experiment: str, row: dict[str, Any], ordinal: int) -> int:
    request_id = row.get("request_id", ordinal)
    payload = f"{SPLIT_VERSION}:{experiment}:{request_id}".encode()
    digest = hashlib.sha256(payload).digest()
    return digest[0] & 1


def selected_records(experiment: str, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = [
        row
        for ordinal, row in enumerate(records)
        if split_bucket(experiment, row, ordinal) == 0
    ]
    if not selected and records:
        midpoint = max(1, len(records) // 2)
        selected = records[:midpoint]
    return selected


def load_reference_metrics(path: Path) -> dict[str, set[str]]:
    if not path.exists():
        return {}
    reference = json.loads(path.read_text())
    return {
        name: set(spec.get("metrics", {}))
        for name, spec in reference.get("experiments", {}).items()
    }


def discover_experiment_dirs(outputs: Path, experiment_name: str | None) -> dict[str, Path]:
    flat_name = experiment_name or outputs.name
    if flat_name in ("", "."):
        flat_name = "custom_trace"

    experiment_dirs: dict[str, Path] = {}
    if outputs.exists() and any(path.suffix == ".json" for path in outputs.iterdir() if path.is_file()):
        experiment_dirs[flat_name] = outputs
    if outputs.exists():
        for child in sorted(outputs.iterdir()):
            if child.is_dir() and any(path.suffix == ".json" for path in child.iterdir() if path.is_file()):
                experiment_dirs[child.name] = child
    return experiment_dirs


def method_files(outdir: Path) -> dict[str, Path]:
    return {
        path.stem: path
        for path in sorted(outdir.glob("*.json"))
        if not path.name.startswith("_")
    }


def aggregate_metrics(path: Path, experiment: str) -> dict[str, float]:
    data = json.loads(path.read_text())
    records = data.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError(f"{path} does not contain non-empty per-request records")

    rows = selected_records(experiment, records)
    if not rows:
        raise ValueError(f"{path} has no records selected for the {SLICE} slice")

    slo_flags = [bool(row.get("slo_met", False)) for row in rows]
    success_fraction = sum(1 for flag in slo_flags if flag) / len(rows)
    offered_rps = float(data.get("offered_rps", 0.0))

    return {
        "effective_rps": offered_rps * success_fraction,
        "slo_violation_rate": 1.0 - success_fraction,
        "ttft_p95_ms": percentile([float(row["ttft_ms"]) for row in rows], 95),
        "ttft_p99_ms": percentile([float(row["ttft_ms"]) for row in rows], 99),
        "tpot_p95_ms": percentile([float(row["max_tpot_ms"]) for row in rows], 95),
        "tpot_p99_ms": percentile([float(row["max_tpot_ms"]) for row in rows], 99),
        "envelope_tpot_p95_ms": percentile(
            [float(row.get("max_envelope_tpot_ms", row["max_tpot_ms"])) for row in rows],
            95,
        ),
        "envelope_tpot_p99_ms": percentile(
            [float(row.get("max_envelope_tpot_ms", row["max_tpot_ms"])) for row in rows],
            99,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs", type=Path, default=Path("outputs"))
    parser.add_argument("--scores", type=Path, default=Path("scoring/scores_train.json"))
    parser.add_argument(
        "--experiment-name",
        default=None,
        help="Experiment name to use when --outputs contains method JSON files directly.",
    )
    args = parser.parse_args()

    reference_metrics = load_reference_metrics(Path("scoring/reference.json"))
    experiment_dirs = discover_experiment_dirs(args.outputs, args.experiment_name)

    experiments: dict[str, dict[str, Any]] = {}
    for experiment, outdir in experiment_dirs.items():
        results: dict[str, dict[str, float]] = {}
        for method_name, path in method_files(outdir).items():
            metrics = aggregate_metrics(path, experiment)
            if experiment in reference_metrics:
                metrics = {k: v for k, v in metrics.items() if k in reference_metrics[experiment]}
            results[method_name] = metrics
        if results:
            experiments[experiment] = {
                "slice": SLICE,
                "split": {"type": SPLIT_VERSION, "bucket": 0, "buckets": 2},
                "results": results,
            }

    if not experiments:
        raise SystemExit(f"No output JSON files found in {args.outputs}")

    scores = {
        "_meta": {"slice": SLICE, "split": SPLIT_VERSION},
        "experiments": {key: experiments[key] for key in sorted(experiments)},
    }
    args.scores.parent.mkdir(parents=True, exist_ok=True)
    args.scores.write_text(json.dumps(scores, indent=2))
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
