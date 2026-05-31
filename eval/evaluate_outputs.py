"""Evaluate scheduler output artifacts.

This module intentionally does not import from method/. It reads JSON artifacts
produced by any candidate method and writes scoring/scores.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


METHOD_FILES = {
    "vllm_vanilla": "vllm_vanilla.json",
    "vllm_sarathi": "vllm_sarathi.json",
    "fairbatching": "fairbatching.json",
    "fairbatching_pab": "fairbatching_pab.json",
}


def load_metrics(path: Path) -> dict[str, float]:
    data = json.loads(path.read_text())
    return {
        "effective_rps": float(data["effective_rps"]),
        "slo_violation_rate": float(data["slo_violation_rate"]),
        "ttft_p99_ms": float(data["ttft_p99_ms"]),
        "tpot_p99_ms": float(data["tpot_p99_ms"]),
        "envelope_tpot_p99_ms": float(data.get("envelope_tpot_p99_ms", data["tpot_p99_ms"])),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs", type=Path, default=Path("outputs"))
    parser.add_argument("--scores", type=Path, default=Path("scoring/scores.json"))
    parser.add_argument(
        "--experiment-name",
        default=None,
        help="Experiment name to use when --outputs contains method JSON files directly.",
    )
    args = parser.parse_args()
    reference_path = Path("scoring/reference.json")
    reference_metrics = {}
    if reference_path.exists():
        reference = json.loads(reference_path.read_text())
        reference_metrics = {
            name: set(spec.get("metrics", {}))
            for name, spec in reference.get("experiments", {}).items()
        }

    flat_name = args.experiment_name or args.outputs.name
    if flat_name in ("", "."):
        flat_name = "custom_trace"
    experiment_dirs = {}
    if any((args.outputs / filename).exists() for filename in METHOD_FILES.values()):
        experiment_dirs[flat_name] = args.outputs
    for child in sorted(args.outputs.iterdir()) if args.outputs.exists() else []:
        if child.is_dir():
            experiment_dirs[child.name] = child

    experiments = {}
    for experiment, outdir in experiment_dirs.items():
        results = {}
        for method, filename in METHOD_FILES.items():
            path = outdir / filename
            if path.exists():
                metrics = load_metrics(path)
                if experiment in reference_metrics:
                    metrics = {k: v for k, v in metrics.items() if k in reference_metrics[experiment]}
                results[method] = metrics
        if results:
            experiments[experiment] = {"results": results}

    if not experiments:
        raise SystemExit(f"No output JSON files found in {args.outputs}")

    scores = {
        "experiments": {
            key: experiments[key]
            for key in sorted(experiments)
        }
    }
    args.scores.parent.mkdir(parents=True, exist_ok=True)
    args.scores.write_text(json.dumps(scores, indent=2))
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
