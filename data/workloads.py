"""Deterministic workload definitions for FairBatching reproduction.

The original production traces are not bundled with the paper. These trace
generators preserve the paper's reported burstiness and token length regimes
while remaining small enough to ship in git.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class WorkloadSpec:
    name: str
    requests: int
    avg_prompt: int
    p90_prompt: int
    avg_output: int
    p90_output: int
    ttft_slo_ms: float
    tpot_slo_ms: float
    default_rps: float


SPECS = {
    "qwen_synthetic": WorkloadSpec(
        name="qwen_synthetic",
        requests=180,
        avg_prompt=892,
        p90_prompt=1776,
        avg_output=377,
        p90_output=742,
        ttft_slo_ms=500.0,
        tpot_slo_ms=50.0,
        default_rps=2.5,
    ),
    "burstgpt_synthetic": WorkloadSpec(
        name="burstgpt_synthetic",
        requests=180,
        avg_prompt=688,
        p90_prompt=1599,
        avg_output=237,
        p90_output=470,
        ttft_slo_ms=500.0,
        tpot_slo_ms=50.0,
        default_rps=3.0,
    ),
    "azure_synthetic": WorkloadSpec(
        name="azure_synthetic",
        requests=180,
        avg_prompt=1604,
        p90_prompt=3561,
        avg_output=114,
        p90_output=392,
        ttft_slo_ms=2000.0,
        tpot_slo_ms=50.0,
        default_rps=1.35,
    ),
}


def generate_decode_slack_burst() -> list[dict]:
    """A compact version of the paper's Figure 1/2 unfairness pattern.

    Long decode requests arrive first and accumulate TPOT slack. A prompt burst
    then arrives while those decodes are still active.
    """
    rows: list[dict] = []
    rid = 0
    for i in range(20):
        rows.append(
            {
                "request_id": rid,
                "arrival_ms": i * 8.0,
                "prompt_len": 96,
                "output_len": 650,
                "ttft_slo_ms": 500.0,
                "tpot_slo_ms": 50.0,
            }
        )
        rid += 1
    for wave, base in enumerate([2200.0, 2400.0, 2600.0]):
        for i in range(16):
            rows.append(
                {
                    "request_id": rid,
                    "arrival_ms": base + i * 3.0,
                    "prompt_len": 850 + (i % 5) * 90,
                    "output_len": 80,
                    "ttft_slo_ms": 500.0,
                    "tpot_slo_ms": 50.0,
                }
            )
            rid += 1
    return rows


def _lognormal_params(mean: float, p90: float) -> tuple[float, float]:
    z90 = 1.2815515655446004
    # Solve ln(p90)=mu+z*sigma and ln(mean)=mu+sigma^2/2.
    ratio = max(p90 / mean, 1.01)
    sigma = z90 - math.sqrt(max(z90 * z90 - 2.0 * math.log(ratio), 0.01))
    sigma = max(0.25, min(1.2, sigma))
    mu = math.log(mean) - sigma * sigma / 2.0
    return mu, sigma


def generate_trace(
    spec: WorkloadSpec,
    seed: int = 7,
    rps: float | None = None,
    ttft_slo_ms: float | None = None,
    tpot_slo_ms: float | None = None,
) -> list[dict]:
    rng = random.Random(seed)
    rps = spec.default_rps if rps is None else rps
    prompt_mu, prompt_sigma = _lognormal_params(spec.avg_prompt, spec.p90_prompt)
    output_mu, output_sigma = _lognormal_params(spec.avg_output, spec.p90_output)

    now = 0.0
    rows: list[dict] = []
    for idx in range(spec.requests):
        if idx % 36 < 18:
            interarrival = rng.expovariate(rps * 2.8)
        elif idx % 36 < 30:
            interarrival = rng.expovariate(rps * 0.55)
        else:
            interarrival = rng.uniform(0.6 / rps, 1.4 / rps)
        now += interarrival * 1000.0
        prompt = int(max(32, min(4096, rng.lognormvariate(prompt_mu, prompt_sigma))))
        output = int(max(16, min(1024, rng.lognormvariate(output_mu, output_sigma))))
        rows.append(
            {
                "request_id": idx,
                "arrival_ms": round(now, 3),
                "prompt_len": prompt,
                "output_len": output,
                "ttft_slo_ms": spec.ttft_slo_ms if ttft_slo_ms is None else ttft_slo_ms,
                "tpot_slo_ms": spec.tpot_slo_ms if tpot_slo_ms is None else tpot_slo_ms,
            }
        )
    return rows


def write_trace(
    name: str,
    output: Path,
    seed: int = 7,
    rps: float | None = None,
    ttft_slo_ms: float | None = None,
    tpot_slo_ms: float | None = None,
) -> None:
    if name == "decode_slack_burst":
        rows = generate_decode_slack_burst()
    else:
        rows = generate_trace(
            SPECS[name],
            seed=seed,
            rps=rps,
            ttft_slo_ms=ttft_slo_ms,
            tpot_slo_ms=tpot_slo_ms,
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", choices=sorted([*SPECS, "decode_slack_burst"]), default="qwen_synthetic")
    parser.add_argument("--output", type=Path, default=Path("data/qwen_synthetic.csv"))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--rps", type=float, default=None)
    parser.add_argument("--ttft-slo-ms", type=float, default=None)
    parser.add_argument("--tpot-slo-ms", type=float, default=None)
    args = parser.parse_args()
    write_trace(args.name, args.output, args.seed, args.rps, args.ttft_slo_ms, args.tpot_slo_ms)


if __name__ == "__main__":
    main()
