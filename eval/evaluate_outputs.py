"""Evaluate scheduler output artifacts.

This module intentionally does not import from method/. It reads JSON artifacts
produced by any candidate method, re-aggregates their per-request records, and
writes scoring/scores.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def percentile(values: list[float], pct: float) -> float:
    values = sorted(values)
    if not values:
        return 0.0
    idx = (len(values) - 1) * pct / 100.0
    lo = int(idx)
    hi = min(lo + 1, len(values) - 1)
    frac = idx - lo
    return values[lo] * (1 - frac) + values[hi] * frac


def load_metrics(path: Path) -> dict[str, float]:
    data = json.loads(path.read_text())
    records = data.get("records")
    if isinstance(records, list) and records:
        ttft_values = [float(row["ttft_ms"]) for row in records]
        tpot_values = [float(row["max_tpot_ms"]) for row in records]
        envelope_values = [
            float(row.get("max_envelope_tpot_ms", row["max_tpot_ms"]))
            for row in records
        ]
        ttft_flags = [
            ttft <= float(row.get("ttft_slo_ms", data.get("ttft_slo_ms", 500.0)))
            for ttft, row in zip(ttft_values, records)
        ]
        envelope_flags = [
            envelope <= float(row.get("tpot_slo_ms", data.get("tpot_slo_ms", 50.0)))
            for envelope, row in zip(envelope_values, records)
        ]
        slo_flags = [
            ttft_ok and tpot_ok
            for ttft_ok, tpot_ok in zip(ttft_flags, envelope_flags)
        ]
        arrivals = [float(row["arrival_ms"]) for row in records if "arrival_ms" in row]
        if len(arrivals) >= 2:
            arrival_span_s = max(1e-9, (max(arrivals) - min(arrivals)) / 1000.0)
            offered_rps = len(records) / arrival_span_s
        else:
            offered_rps = float(data.get("offered_rps", 0.0))

        success_fraction = sum(1 for flag in slo_flags if flag) / len(records)
        ttft_success = sum(1 for flag in ttft_flags if flag) / len(records)
        envelope_success = sum(1 for flag in envelope_flags if flag) / len(records)
        return {
            "effective_rps": offered_rps * success_fraction,
            "slo_violation_rate": 1.0 - success_fraction,
            "ttft_slo_violation_rate": 1.0 - ttft_success,
            "envelope_tpot_slo_violation_rate": 1.0 - envelope_success,
            "ttft_p95_ms": percentile(ttft_values, 95),
            "ttft_p99_ms": percentile(ttft_values, 99),
            "tpot_p95_ms": percentile(tpot_values, 95),
            "tpot_p99_ms": percentile(tpot_values, 99),
            "envelope_tpot_p95_ms": percentile(envelope_values, 95),
            "envelope_tpot_p99_ms": percentile(envelope_values, 99),
        }

    raise ValueError(f"{path} does not contain non-empty per-request records")


def method_files(outdir: Path) -> dict[str, Path]:
    return {
        path.stem: path
        for path in sorted(outdir.glob("*.json"))
        if not path.name.startswith("_")
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
    reference_types = {}
    if reference_path.exists():
        reference = json.loads(reference_path.read_text())
        reference_metrics = {
            name: set(spec.get("metrics", {}))
            for name, spec in reference.get("experiments", {}).items()
        }
        reference_types = {
            name: {
                method: result.get("type")
                for method, result in spec.get("results", {}).items()
                if isinstance(result, dict)
            }
            for name, spec in reference.get("experiments", {}).items()
        }

    flat_name = args.experiment_name or args.outputs.name
    if flat_name in ("", "."):
        flat_name = "custom_trace"
    experiment_dirs = {}
    if method_files(args.outputs):
        experiment_dirs[flat_name] = args.outputs
    for child in sorted(args.outputs.iterdir()) if args.outputs.exists() else []:
        if child.is_dir():
            experiment_dirs[child.name] = child

    experiments = {}
    for experiment, outdir in experiment_dirs.items():
        results = {}
        for method, path in method_files(outdir).items():
            results[method] = load_metrics(path)
        baseline_methods = [
            method
            for method in results
            if reference_types.get(experiment, {}).get(method) == "baseline"
        ]
        if not baseline_methods:
            baseline_methods = [
                method
                for method in results
                if method.startswith("vllm_") or "baseline" in method
            ]
        if baseline_methods:
            best_baseline = max(results[m]["effective_rps"] for m in baseline_methods)
            if best_baseline > 0:
                for metrics in results.values():
                    metrics["goodput_improvement_pct"] = (
                        metrics["effective_rps"] / best_baseline - 1.0
                    ) * 100.0
        for method, metrics in list(results.items()):
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
