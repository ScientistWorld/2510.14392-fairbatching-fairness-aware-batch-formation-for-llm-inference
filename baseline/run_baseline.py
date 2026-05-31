"""Run baseline schedulers using the shared simulator policies."""

from __future__ import annotations

import argparse
from pathlib import Path

from method.simulator import ModelParams, run_simulation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, default=Path("data/qwen_synthetic.csv"))
    parser.add_argument("--policy", choices=["vllm_vanilla", "vllm_sarathi"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run_simulation(args.trace, args.policy, args.output, ModelParams())


if __name__ == "__main__":
    main()
