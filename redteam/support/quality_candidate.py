"""Real scheduling variants. Execution cost and workload remain unchanged."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from data.serving_sim import (
    BatchItem, ModelParams, one_decode, all_waiting_prefill,
    run_simulation, token_time,
)
from method.simulator import FairBatchingScheduler


class CappedFair(FairBatchingScheduler):
    def __init__(self, params, cap=100.0, raw=False):
        super().__init__(params)
        self.cap = cap
        self.raw = raw

    def select(self, active, now_ms):
        # Preserve the three stages, but limit long batches that hurt raw TPOT.
        candidates = [r for r in active if not r.done]
        if not candidates:
            return []
        decodes = [r for r in candidates if r.is_decode]
        min_tpot = min(r.tpot_slo_ms for r in (decodes or candidates))
        budget = max(min(r.slack(now_ms) for r in (decodes or candidates)), min_tpot)
        budget = min(budget, self.cap)
        if self.raw and decodes:
            raw_slacks = [r.output_times[0] + r.tpot_slo_ms * r.generated - now_ms for r in decodes]
            budget = min(budget, max(min(raw_slacks), min_tpot))
        remaining = max(0.0, budget - self.params.a)
        tokens = self.params.max_token_budget
        urgent = [r for r in decodes if r.slack(now_ms) < budget + min_tpot]
        if self.raw:
            urgent = [r for r in decodes if min(r.slack(now_ms), r.output_times[0] + r.tpot_slo_ms * r.generated - now_ms) < budget + min_tpot]
        prefills = [r for r in candidates if r.is_prefill]
        nonurgent = [r for r in decodes if r not in urgent]
        batch = []
        for group in (urgent, prefills, nonurgent):
            for req in sorted(group, key=lambda r: r.slack(now_ms)):
                item = one_decode(req) if req.is_decode else all_waiting_prefill(req)
                cost = token_time(item, self.params)
                if item.new_tokens <= tokens and cost <= remaining:
                    batch.append(item)
                    tokens -= item.new_tokens
                    remaining -= cost
                elif req.is_prefill and tokens and self.params.c * req.context_len <= remaining:
                    chunk = min(tokens, item.new_tokens, int((remaining - self.params.c * req.context_len) / self.params.b))
                    if chunk > 0:
                        batch.append(BatchItem(req, chunk, "prefill"))
                        tokens -= chunk
                        remaining -= self.params.b * chunk + self.params.c * req.context_len
        if not batch and decodes:
            return [one_decode(min(decodes, key=lambda r: r.slack(now_ms)))]
        return batch


class FixedBudget:
    """Ordinary fixed token budget; no deadline or slack calculations."""
    def __init__(self, params, budget=1000, order="decode"):
        self.params, self.budget, self.order = params, budget, order

    def select(self, active, now_ms):
        prefills = sorted((r for r in active if r.is_prefill), key=lambda r: r.arrival_ms)
        decodes = sorted((r for r in active if r.is_decode), key=lambda r: r.arrival_ms)
        batch, tokens = [], self.budget
        groups = (decodes, prefills) if self.order == "decode" else (prefills, decodes)
        for group in groups:
            for req in group:
                if tokens <= 0:
                    break
                if req.is_decode:
                    batch.append(one_decode(req))
                    tokens -= 1
                else:
                    count = min(tokens, req.prompt_len - req.prefill_done)
                    batch.append(BatchItem(req, count, "prefill"))
                    tokens -= count
        return batch


class SerialFIFO:
    """Deliberately weak: serialize requests and remove all batching."""
    def select(self, active, now_ms):
        req = min(active, key=lambda r: r.arrival_ms)
        return [one_decode(req) if req.is_decode else all_waiting_prefill(req)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=["capped", "raw", "fixed", "weak"], required=True)
    parser.add_argument("--cap", type=float, default=100.)
    parser.add_argument("--budget", type=int, default=1000)
    parser.add_argument("--order", default="decode")
    parser.add_argument("--trace", type=Path, default=Path("data/qwen_synthetic.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs/single_node_qwen_trace/fairbatching_pab.json"))
    args = parser.parse_args()
    params = ModelParams()
    if args.kind in ("capped", "raw"):
        scheduler = CappedFair(params, args.cap, args.kind == "raw")
    elif args.kind == "fixed":
        scheduler = FixedBudget(params, args.budget, args.order)
    else:
        scheduler = SerialFIFO()
    result = run_simulation(args.trace, args.kind, scheduler, args.output, params)
    print(json.dumps({k: v for k, v in result.items() if k != "records"}, indent=2))


if __name__ == "__main__":
    main()
