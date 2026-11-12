"""Evaluator-owned copies of the paper's baselines (train split).

The goodput gains in the scores are measured against these frozen copies, not
against whatever ``baseline/`` contains, so editing ``baseline/`` cannot move
the comparison point.  Logic is identical to ``baseline/schedulers.py``.
"""

from __future__ import annotations

from data.sched_client import Policy, RequestView

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
    def select(self, active: list[RequestView], now_ms: float) -> list[tuple[int, int]]:
        tpot = min(r.tpot_slo_ms for r in active)
        budget = min(SARATHI_BUDGET.get(tpot, SARATHI_BUDGET[50.0]), self.profile.max_batch_tokens)
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


FROZEN_BASELINES = {"vllm_vanilla": VllmVanilla, "vllm_sarathi": VllmSarathi}
