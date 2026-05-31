"""Trace-driven LLM serving scheduler simulator.

This implements FairBatching's envelope-deadline SLO tracking, adaptive
time-based batch budget, and three-stage batch construction from Algorithm 1.
The execution model is the paper's linear step-time estimator:

    batch_time = a + b * total_new_tokens + c * total_context
"""

from __future__ import annotations

import argparse
import csv
import heapq
import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Request:
    request_id: int
    arrival_ms: float
    prompt_len: int
    output_len: int
    ttft_slo_ms: float
    tpot_slo_ms: float
    prefill_done: int = 0
    generated: int = 0
    output_times: list[float] = field(default_factory=list)
    rejected: bool = False

    @property
    def is_prefill(self) -> bool:
        return self.prefill_done < self.prompt_len

    @property
    def is_decode(self) -> bool:
        return self.prefill_done >= self.prompt_len and self.generated < self.output_len

    @property
    def done(self) -> bool:
        return self.rejected or self.generated >= self.output_len

    @property
    def context_len(self) -> int:
        return self.prefill_done + self.generated

    def next_deadline(self) -> float:
        if self.is_prefill:
            return self.arrival_ms + self.ttft_slo_ms
        return self.arrival_ms + self.ttft_slo_ms + self.tpot_slo_ms * self.generated

    def slack(self, now_ms: float) -> float:
        return self.next_deadline() - now_ms


@dataclass(frozen=True)
class ModelParams:
    a: float = 8.0
    b: float = 0.025
    c: float = 0.0008
    max_token_budget: int = 6000
    sarathi_token_budget: int = 1000


@dataclass(frozen=True)
class BatchItem:
    request: Request
    new_tokens: int
    kind: str


def estimate_time(items: list[BatchItem], params: ModelParams) -> float:
    if not items:
        return 0.0
    return params.a + params.b * sum(x.new_tokens for x in items) + params.c * sum(x.request.context_len for x in items)


def token_time(item: BatchItem, params: ModelParams) -> float:
    return params.b * item.new_tokens + params.c * item.request.context_len


def load_trace(path: Path) -> list[Request]:
    rows = []
    with path.open() as f:
        for row in csv.DictReader(f):
            rows.append(
                Request(
                    request_id=int(row["request_id"]),
                    arrival_ms=float(row["arrival_ms"]),
                    prompt_len=int(row["prompt_len"]),
                    output_len=int(row["output_len"]),
                    ttft_slo_ms=float(row["ttft_slo_ms"]),
                    tpot_slo_ms=float(row["tpot_slo_ms"]),
                )
            )
    return rows


def _all_waiting_prefill(req: Request) -> BatchItem:
    return BatchItem(req, req.prompt_len - req.prefill_done, "prefill")


def _one_decode(req: Request) -> BatchItem:
    return BatchItem(req, 1, "decode")


class Scheduler:
    def __init__(self, params: ModelParams):
        self.params = params

    def select(self, active: list[Request], now_ms: float) -> list[BatchItem]:
        raise NotImplementedError


class FairBatchingScheduler(Scheduler):
    """FairBatching Algorithm 1: urgent decodes, prefills, non-urgent decodes."""

    def select(self, active: list[Request], now_ms: float) -> list[BatchItem]:
        candidates = [r for r in active if not r.done]
        if not candidates:
            return []
        decodes = [r for r in candidates if r.is_decode]
        min_tpot = min((r.tpot_slo_ms for r in decodes), default=min(r.tpot_slo_ms for r in candidates))
        min_slack = min((r.slack(now_ms) for r in decodes), default=min(r.slack(now_ms) for r in candidates))
        init_time_budget = max(min_slack, min_tpot)
        remaining_time = max(0.0, init_time_budget - self.params.a)
        token_budget = self.params.max_token_budget

        urgent_decode: list[Request] = []
        prefill: list[Request] = []
        nonurgent_decode: list[Request] = []
        for req in candidates:
            if req.is_decode and req.slack(now_ms) < init_time_budget + 8.0 * min_tpot:
                urgent_decode.append(req)
            elif req.is_prefill:
                prefill.append(req)
            elif req.is_decode:
                nonurgent_decode.append(req)

        groups = [
            sorted(urgent_decode, key=lambda r: r.slack(now_ms)),
            sorted(prefill, key=lambda r: r.slack(now_ms)),
            sorted(nonurgent_decode, key=lambda r: r.slack(now_ms)),
        ]
        batch: list[BatchItem] = []
        for group in groups:
            for req in group:
                if token_budget <= 0:
                    return batch
                item = _one_decode(req) if req.is_decode else _all_waiting_prefill(req)
                full_cost = token_time(item, self.params)
                if item.new_tokens <= token_budget and full_cost <= remaining_time:
                    batch.append(item)
                    token_budget -= item.new_tokens
                    remaining_time -= full_cost
                elif req.is_prefill and token_budget > 0 and self.params.c * req.context_len <= remaining_time:
                    chunk = int((remaining_time - self.params.c * req.context_len) / self.params.b)
                    chunk = max(0, min(chunk, token_budget, req.prompt_len - req.prefill_done))
                    if chunk > 0:
                        batch.append(BatchItem(req, chunk, "prefill"))
                        token_budget -= chunk
                        remaining_time -= self.params.b * chunk + self.params.c * req.context_len
        if not batch and decodes:
            # Graceful fallback under extreme overload, matching the paper's discussion.
            return [_one_decode(min(decodes, key=lambda r: r.slack(now_ms)))]
        return batch


class SarathiScheduler(Scheduler):
    """Decode-prioritizing stall-free batching baseline."""

    def select(self, active: list[Request], now_ms: float) -> list[BatchItem]:
        token_budget = self.params.sarathi_token_budget
        remaining_time = min((r.tpot_slo_ms for r in active if r.is_decode), default=50.0) - self.params.a
        batch: list[BatchItem] = []
        for req in sorted([r for r in active if r.is_decode], key=lambda r: r.request_id):
            item = _one_decode(req)
            if token_budget >= 1 and token_time(item, self.params) <= remaining_time:
                batch.append(item)
                token_budget -= 1
                remaining_time -= token_time(item, self.params)
        for req in sorted([r for r in active if r.is_prefill], key=lambda r: r.arrival_ms):
            if token_budget <= 0:
                break
            chunk = min(req.prompt_len - req.prefill_done, token_budget)
            if self.params.b * chunk + self.params.c * req.context_len > remaining_time:
                chunk = int(max(0, (remaining_time - self.params.c * req.context_len) / self.params.b))
            if chunk > 0:
                batch.append(BatchItem(req, chunk, "prefill"))
                token_budget -= chunk
                remaining_time -= self.params.b * chunk + self.params.c * req.context_len
        return batch


class VanillaScheduler(Scheduler):
    """Prefill-prioritizing continuous batching baseline similar to default vLLM."""

    def select(self, active: list[Request], now_ms: float) -> list[BatchItem]:
        token_budget = self.params.max_token_budget
        batch: list[BatchItem] = []
        for req in sorted([r for r in active if r.is_prefill], key=lambda r: r.arrival_ms):
            if token_budget <= 0:
                break
            chunk = min(req.prompt_len - req.prefill_done, token_budget)
            batch.append(BatchItem(req, chunk, "prefill"))
            token_budget -= chunk
        for req in sorted([r for r in active if r.is_decode], key=lambda r: r.request_id):
            if token_budget <= 0:
                break
            batch.append(_one_decode(req))
            token_budget -= 1
        return batch


SCHEDULERS = {
    "fairbatching": FairBatchingScheduler,
    "fairbatching_pab": FairBatchingScheduler,
    "vllm_sarathi": SarathiScheduler,
    "vllm_vanilla": VanillaScheduler,
}


def prefill_admission_budget(active: list[Request], now_ms: float, incoming: Request, params: ModelParams) -> float:
    """Appendix A PAB estimate for a new request's TTFT window."""
    considered = [r for r in active if not r.done]
    if not considered:
        return float("inf")
    ttft = incoming.ttft_slo_ms
    tpot = incoming.tpot_slo_ms
    min_slack = min(r.slack(now_ms) for r in considered)
    n_batches = max(0.0, (ttft - min_slack) / tpot) + 1.0
    reserved = n_batches * params.a
    for req in considered:
        n_steps = max(0.0, (ttft - req.slack(now_ms)) / tpot)
        reserved += n_steps * (params.b + req.context_len * params.c)
    capacity = (ttft - reserved) / (params.b + params.c)
    existing_prefill = sum(r.prompt_len - r.prefill_done for r in considered if r.is_prefill)
    return capacity - existing_prefill


def run_simulation(trace_path: Path, policy: str, output_path: Path, params: ModelParams) -> dict:
    requests = load_trace(trace_path)
    pending = [(r.arrival_ms, r.request_id, r) for r in requests]
    heapq.heapify(pending)
    active: list[Request] = []
    now_ms = pending[0][0] if pending else 0.0
    scheduler = SCHEDULERS[policy](params)
    steps = 0
    prefill_tokens = 0
    decode_tokens = 0
    total_step_time = 0.0

    while pending or any(not r.done for r in active):
        while pending and pending[0][0] <= now_ms:
            req = heapq.heappop(pending)[2]
            if policy == "fairbatching_pab" and prefill_admission_budget(active, now_ms, req, params) < 1.2 * req.prompt_len:
                req.rejected = True
            active.append(req)
        active = [r for r in active if not r.done]
        if not active:
            if not pending:
                break
            now_ms = pending[0][0]
            continue
        batch = scheduler.select(active, now_ms)
        if not batch:
            now_ms += 0.1
            continue
        duration = estimate_time(batch, params)
        end_ms = now_ms + duration
        for item in batch:
            req = item.request
            if item.kind == "prefill":
                req.prefill_done += item.new_tokens
                prefill_tokens += item.new_tokens
                if req.prefill_done >= req.prompt_len and req.generated == 0:
                    req.generated = 1
                    req.output_times.append(end_ms)
                    decode_tokens += 1
            else:
                req.generated += 1
                req.output_times.append(end_ms)
                decode_tokens += 1
        now_ms = end_ms
        steps += 1
        total_step_time += duration

    rows = []
    for req in sorted(requests, key=lambda r: r.request_id):
        if req.output_times:
            ttft = req.output_times[0] - req.arrival_ms
            if len(req.output_times) > 1:
                tpots = [
                    (req.output_times[i] - req.output_times[0]) / i
                    for i in range(1, len(req.output_times))
                ]
                max_tpot = max(tpots)
            else:
                max_tpot = 0.0
            slo_met = ttft <= req.ttft_slo_ms and max_tpot <= req.tpot_slo_ms
        else:
            ttft = 2.0 * req.ttft_slo_ms
            max_tpot = 2.0 * req.tpot_slo_ms
            slo_met = False
        rows.append(
            {
                "request_id": req.request_id,
                "arrival_ms": req.arrival_ms,
                "prompt_len": req.prompt_len,
                "output_len": req.output_len,
                "ttft_ms": ttft,
                "max_tpot_ms": max_tpot,
                "ttft_slo_ms": req.ttft_slo_ms,
                "tpot_slo_ms": req.tpot_slo_ms,
                "slo_met": slo_met,
            }
        )

    last_output = max((r.output_times[-1] for r in requests if r.output_times), default=requests[-1].arrival_ms)
    makespan_s = max(1e-9, (last_output - requests[0].arrival_ms) / 1000.0)
    summary = {
        "policy": policy,
        "trace": str(trace_path),
        "requests": len(requests),
        "steps": steps,
        "prefill_tokens": prefill_tokens,
        "decode_tokens": decode_tokens,
        "mean_step_ms": total_step_time / max(steps, 1),
        "effective_rps": sum(1 for r in rows if r["slo_met"]) / makespan_s,
        "slo_violation_rate": 1.0 - sum(1 for r in rows if r["slo_met"]) / len(rows),
        "ttft_p95_ms": percentile([r["ttft_ms"] for r in rows], 95),
        "ttft_p99_ms": percentile([r["ttft_ms"] for r in rows], 99),
        "tpot_p95_ms": percentile([r["max_tpot_ms"] for r in rows], 95),
        "tpot_p99_ms": percentile([r["max_tpot_ms"] for r in rows], 99),
        "records": rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2))
    return summary


def percentile(values: list[float], pct: float) -> float:
    values = sorted(values)
    if not values:
        return 0.0
    idx = (len(values) - 1) * pct / 100.0
    lo = int(idx)
    hi = min(lo + 1, len(values) - 1)
    frac = idx - lo
    return values[lo] * (1 - frac) + values[hi] * frac


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, default=Path("data/qwen_synthetic.csv"))
    parser.add_argument("--policy", choices=sorted(SCHEDULERS), default="fairbatching")
    parser.add_argument("--output", type=Path, default=Path("outputs/fairbatching.json"))
    args = parser.parse_args()
    summary = run_simulation(args.trace, args.policy, args.output, ModelParams())
    print(json.dumps({k: v for k, v in summary.items() if k != "records"}, indent=2))


if __name__ == "__main__":
    main()
