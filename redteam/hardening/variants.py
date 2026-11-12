"""Hardening-audit scheduler variants (iteration 11).

Honest modifications of FairBatching(-PAB) used to check that the gym ranks
genuinely better schedulers above the reference and weaker ones below it, plus
the brute-force static-budget scheduler the paper's adaptive budget replaces.
Run as ``python redteam/hardening/variants.py --variant NAME [...]``.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from data.sched_client import Policy, serve  # noqa: E402
from method.fairbatching import FairBatching  # noqa: E402


class TunableFB(FairBatching):
    """Algorithm 1 with optional knobs (all off = the reference)."""

    def __init__(self, use_pab=True, margin=0.0, shed=False, srpt=False, fixed_budget=False,
                 exact_cost=False, cap_ms=None):
        super().__init__(use_pab)
        self.margin = margin
        self.shed = shed
        self.srpt = srpt
        self.fixed_budget = fixed_budget
        self.exact_cost = exact_cost
        self.cap_ms = cap_ms

    # --- cost model -------------------------------------------------------
    def item_cost(self, r, n):
        p = self.profile
        if self.exact_cost:  # uses the simulator's published hardware formula
            from data.serving_sim import HARDWARE as H
            extra = H["d"] * n * (r.prefill_done + n / 2.0) if r.is_prefill else 0.0
            return H["b"] * n + H["c"] * r.context_len + extra
        return p.b * n + p.c * r.context_len

    def max_chunk(self, r, budget):
        p = self.profile
        if self.exact_cost:
            from data.serving_sim import HARDWARE as H
            qa, qb, qc = H["d"] / 2.0, H["b"] + H["d"] * r.prefill_done, H["c"] * r.context_len - budget
            if qc > 0:
                return 0
            return int((-qb + math.sqrt(qb * qb - 4 * qa * qc)) / (2 * qa))
        if p.c * r.context_len > budget:
            return 0
        return int((budget - p.c * r.context_len) / p.b)

    def fixed_overhead(self):
        if self.exact_cost:
            from data.serving_sim import HARDWARE as H
            return H["a"]
        return self.profile.a

    # --- extensions ---------------------------------------------------------
    def drop(self, active, now_ms):
        if not self.shed:
            return ()
        out = []
        for r in active:
            if r.is_prefill and not r.started:
                need = self.fixed_overhead() + self.item_cost(r, r.remaining_prefill)
                if now_ms + need > r.arrival_ms + r.ttft_slo_ms:
                    out.append(r.rid)
        return out

    def select(self, active, now_ms):
        decodes = [r for r in active if r.is_decode]
        if decodes:
            min_tpot = min(r.tpot_slo_ms for r in decodes)
            min_slack = min(r.slack(now_ms) for r in decodes)
            init_time_budget = min_tpot if self.fixed_budget else max(min_slack, min_tpot)
        else:
            min_tpot = min(r.tpot_slo_ms for r in active)
            init_time_budget = min_tpot if self.fixed_budget else math.inf
        if self.cap_ms is not None:
            init_time_budget = min(init_time_budget, self.cap_ms)
        if math.isfinite(init_time_budget):
            init_time_budget *= 1.0 - self.margin
        time_budget = init_time_budget - self.fixed_overhead()
        token_budget = self.profile.max_batch_tokens
        urgent, prefill, nonurgent = [], [], []
        for r in active:
            if r.is_decode and r.slack(now_ms) < init_time_budget + min_tpot:
                urgent.append(r)
            elif r.is_prefill:
                prefill.append(r)
            else:
                nonurgent.append(r)
        key = lambda r: r.slack(now_ms)  # noqa: E731
        pkey = (lambda r: (r.remaining_prefill, r.slack(now_ms))) if self.srpt else key
        batch = []
        for r in sorted(urgent, key=key) + sorted(prefill, key=pkey) + sorted(nonurgent, key=key):
            if token_budget <= 0:
                break
            n = 1 if r.is_decode else r.remaining_prefill
            cost = self.item_cost(r, n)
            if n <= token_budget and cost <= time_budget:
                batch.append((r.rid, n))
                time_budget -= cost
                token_budget -= n
            elif r.is_prefill:
                chunk = min(token_budget, self.max_chunk(r, time_budget), r.remaining_prefill)
                if chunk > 0:
                    batch.append((r.rid, chunk))
                    time_budget -= self.item_cost(r, chunk)
                    token_budget -= chunk
        if not batch and decodes:
            batch.append((min(decodes, key=key).rid, 1))
        return batch


class SerialFifo(Policy):
    """Weak variant: one request at a time, first come first served."""

    def select(self, active, now_ms):
        r = min(active, key=lambda r: (r.arrival_ms, r.rid))
        return [(r.rid, 1 if r.is_decode else min(r.remaining_prefill, self.profile.max_batch_tokens))]


class StaticBudget(Policy):
    """Brute-force alternative to the adaptive budget: stall-free batching with
    a static token budget per TPOT SLO (decode-first, FCFS chunked prefill)."""

    def __init__(self, budgets):
        self.budgets = budgets

    def select(self, active, now_ms):
        tpot = min(r.tpot_slo_ms for r in active)
        budget = min(self.budgets.get(tpot, max(self.budgets.values())), self.profile.max_batch_tokens)
        batch = []
        for r in sorted((r for r in active if r.is_decode), key=lambda r: (r.arrival_ms, r.rid)):
            if budget <= 0:
                break
            batch.append((r.rid, 1))
            budget -= 1
        for r in sorted((r for r in active if r.is_prefill), key=lambda r: (r.arrival_ms, r.rid)):
            if budget <= 0:
                break
            n = min(r.remaining_prefill, budget)
            batch.append((r.rid, n))
            budget -= n
        return batch


VARIANTS = {
    # better (honest refinements)
    "margin_pab": lambda: TunableFB(True, margin=0.06),
    "shed_pab": lambda: TunableFB(True, margin=0.06, shed=True),
    "srpt_shed_pab": lambda: TunableFB(True, margin=0.06, shed=True, srpt=True),
    "exact_cost_pab": lambda: TunableFB(True, margin=0.012, exact_cost=True),
    # weaker
    "fixed_batch_pab": lambda: TunableFB(True, fixed_budget=True),
    "fixed_batch_fb": lambda: TunableFB(False, fixed_budget=True),
    "serial_fifo": SerialFifo,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=sorted(VARIANTS) + ["static_budget"], required=True)
    parser.add_argument("--budgets", default="50:256,100:512", help="tpot_slo:tokens,...")
    args = parser.parse_args()
    if args.variant == "static_budget":
        budgets = {float(k): int(v) for k, v in (x.split(":") for x in args.budgets.split(","))}
        serve(StaticBudget(budgets))
    else:
        serve(VARIANTS[args.variant]())


if __name__ == "__main__":
    main()
