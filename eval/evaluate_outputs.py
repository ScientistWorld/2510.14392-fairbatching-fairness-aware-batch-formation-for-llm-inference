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
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs", type=Path, default=Path("outputs"))
    parser.add_argument("--scores", type=Path, default=Path("scoring/scores.json"))
    args = parser.parse_args()

    results = {}
    for method, filename in METHOD_FILES.items():
        path = args.outputs / filename
        if path.exists():
            results[method] = load_metrics(path)
    if not results:
        raise SystemExit(f"No output JSON files found in {args.outputs}")

    scores = {
        "experiments": {
            "single_node_qwen_trace": {
                "results": results,
            }
        }
    }
    args.scores.parent.mkdir(parents=True, exist_ok=True)
    args.scores.write_text(json.dumps(scores, indent=2))
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
