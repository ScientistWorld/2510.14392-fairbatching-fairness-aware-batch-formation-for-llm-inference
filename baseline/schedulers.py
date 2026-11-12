"""Baseline schedulers from the paper's evaluation, as protocol programs.

* ``vllm_vanilla``: prefill-prioritizing continuous batching with a large
  token budget (vLLM default): waiting prefills are packed first-come
  first-served up to ``max_batch_tokens``, then running decodes fill the rest.
* ``vllm_sarathi``: Sarathi-Serve stall-free batching: every running decode is
  included first, then prefills are chunked first-come first-served into a
  fixed token budget.  The paper tunes the budget per test case; the budgets
  below were chosen by sweeping {128, 160, 192, 256, 384, 512, 768, 1024,
  2048} on the public train workloads (peak goodput) and are fixed per SLO
  setting.

Usage: ``python baseline/schedulers.py --policy {vllm_vanilla,vllm_sarathi}
[--token-budget N]``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.sched_client import Policy, RequestView, serve  # noqa: E402

# Tuned Sarathi token budget per TPOT SLO (ms) -> tokens.
SARATHI_BUDGET = {50.0: 256, 100.0: 512}


class VllmVanilla(Policy):
    def select(self, active: list[RequestView], now_ms: float) -> list[tuple[int, int]]:
        budget = self.profile.max_batch_tokens
        batch = []
        for r in sorted((r for r in active if r.is_prefill), key=lambda r: (r.arrival_ms, r.rid)):
            if budget <= 0:
                break
            n = min(r.remaining_prefill, budget)
            batch.append((r.rid, n))
            budget -= n
        for r in sorted((r for r in active if r.is_decode), key=lambda r: (r.arrival_ms, r.rid)):
            if budget <= 0:
                break
            batch.append((r.rid, 1))
            budget -= 1
        return batch


class VllmSarathi(Policy):
    def __init__(self, token_budget: int | None = None):
        self.token_budget = token_budget

    def budget_for(self, active: list[RequestView]) -> int:
        if self.token_budget is not None:
            return self.token_budget
        tpot = min(r.tpot_slo_ms for r in active)
        return SARATHI_BUDGET.get(tpot, SARATHI_BUDGET[50.0])

    def select(self, active: list[RequestView], now_ms: float) -> list[tuple[int, int]]:
        budget = min(self.budget_for(active), self.profile.max_batch_tokens)
        batch = []
        for r in sorted((r for r in active if r.is_decode), key=lambda r: (r.arrival_ms, r.rid)):
            batch.append((r.rid, 1))
            budget -= 1
        for r in sorted((r for r in active if r.is_prefill), key=lambda r: (r.arrival_ms, r.rid)):
            if budget <= 0:
                break
            n = min(r.remaining_prefill, budget)
            batch.append((r.rid, n))
            budget -= n
        return batch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=["vllm_vanilla", "vllm_sarathi"], required=True)
    parser.add_argument("--token-budget", type=int, default=None)
    args = parser.parse_args()
    if args.policy == "vllm_vanilla":
        serve(VllmVanilla())
    else:
        serve(VllmSarathi(args.token_budget))


if __name__ == "__main__":
    main()
