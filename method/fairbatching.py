"""FairBatching scheduler (Lyu et al., 2025) as a protocol scheduler program.

Implements the paper's computational steps:

* Section 3.1 envelope deadlines: token j of request i is due at
  ``arrival_i + ttft_slo_i + tpot_slo_i * j``; slack is the next token's
  deadline minus now (``RequestView.slack``).
* Section 3.2 adaptive time budget: ``init_time_budget = max(min slack of
  decode requests, min TPOT SLO of decode requests)``; with no decode request
  there is no slack to protect and only the token budget applies.  The batch
  time is estimated with the linear model ``a + b*new_tokens + c*context``
  published by the evaluator.
* Section 3.3 / Algorithm 1 batch formation: urgent decodes (slack <
  init_time_budget + min_tpot_slo), then prefills, then the remaining decodes,
  each group sorted by slack; items are added while ``b*new_tokens +
  c*context`` fits the remaining time budget (``init_time_budget - a``) and
  the token budget, and a prefill that does not fit is chunked to the
  remaining budget.
* Section 3.4 / Appendix A prefill admission budget (``--variant pab``): a new
  request is rejected when the PAB computed over all admitted requests is
  smaller than its prompt length.

Usage: ``python method/fairbatching.py --variant {vanilla,pab}``.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.sched_client import Policy, RequestView, serve  # noqa: E402


class FairBatching(Policy):
    def __init__(self, use_pab: bool = False):
        self.use_pab = use_pab

    # ---------------- Section 3.4 / Appendix A ----------------
    def prefill_admission_budget(self, admitted: list[RequestView], now: float, incoming: RequestView) -> float:
        if not admitted:
            return math.inf
        p = self.profile
        ttft = incoming.ttft_slo_ms
        tpot = incoming.tpot_slo_ms
        slacks = [r.slack(now) for r in admitted]
        n_batches = max(0.0, (ttft - min(slacks)) / tpot) + 1.0
        reserved = n_batches * p.a
        for r, s in zip(admitted, slacks):
            reserved += max(0.0, (ttft - s) / tpot) * (p.b + r.context_len * p.c)
        capacity = (ttft - reserved) / (p.b + p.c)
        existing_prefill = sum(r.remaining_prefill for r in admitted if r.is_prefill)
        return capacity - existing_prefill

    def admit(self, request: RequestView, admitted: list[RequestView], now_ms: float) -> bool:
        if not self.use_pab:
            return True
        return self.prefill_admission_budget(admitted, now_ms, request) >= request.prompt_len

    # ---------------- Sections 3.2-3.3 / Algorithm 1 ----------------
    def select(self, active: list[RequestView], now_ms: float) -> list[tuple[int, int]]:
        p = self.profile
        decodes = [r for r in active if r.is_decode]
        if decodes:
            min_tpot = min(r.tpot_slo_ms for r in decodes)
            min_slack = min(r.slack(now_ms) for r in decodes)
            init_time_budget = max(min_slack, min_tpot)
        else:
            min_tpot = min(r.tpot_slo_ms for r in active)
            init_time_budget = math.inf
        time_budget = init_time_budget - p.a
        token_budget = p.max_batch_tokens

        urgent, prefill, nonurgent = [], [], []
        for r in active:
            if r.is_decode and r.slack(now_ms) < init_time_budget + min_tpot:
                urgent.append(r)
            elif r.is_prefill:
                prefill.append(r)
            else:
                nonurgent.append(r)
        key = lambda r: r.slack(now_ms)  # noqa: E731
        ordered = sorted(urgent, key=key) + sorted(prefill, key=key) + sorted(nonurgent, key=key)

        batch: list[tuple[int, int]] = []
        for r in ordered:
            if token_budget <= 0:
                break
            new_tokens = 1 if r.is_decode else r.remaining_prefill
            cost = p.b * new_tokens + p.c * r.context_len
            if new_tokens <= token_budget and cost <= time_budget:
                batch.append((r.rid, new_tokens))
                time_budget -= cost
                token_budget -= new_tokens
            elif r.is_prefill and p.c * r.context_len <= time_budget:
                room = (time_budget - p.c * r.context_len) / p.b
                chunk = int(min(token_budget, room, r.remaining_prefill))
                if chunk > 0:
                    batch.append((r.rid, chunk))
                    time_budget -= p.b * chunk + p.c * r.context_len
                    token_budget -= chunk
        if not batch and decodes:
            # Under extreme overload fall back to the most urgent decode.
            r = min(decodes, key=key)
            batch.append((r.rid, 1))
        return batch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["vanilla", "pab"], default="pab")
    args = parser.parse_args()
    serve(FairBatching(use_pab=args.variant == "pab"))


if __name__ == "__main__":
    main()
