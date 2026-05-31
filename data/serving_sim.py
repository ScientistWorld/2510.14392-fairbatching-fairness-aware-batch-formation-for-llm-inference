"""Method-agnostic trace-driven serving simulator utilities.

The simulator executes scheduler decisions against a request trace and writes
request-level timing artifacts. Scheduler implementations live outside this
module so evaluation and baselines do not depend on the paper method.
"""

from __future__ import annotations

import csv
import heapq
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol


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


class Scheduler(Protocol):
    def select(self, active: list[Request], now_ms: float) -> list[BatchItem]:
        ...


AdmissionFn = Callable[[list[Request], float, Request, ModelParams], bool]


def estimate_time(items: list[BatchItem], params: ModelParams) -> float:
    if not items:
        return 0.0
    return params.a + params.b * sum(x.new_tokens for x in items) + params.c * sum(
        x.request.context_len for x in items
    )


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


def all_waiting_prefill(req: Request) -> BatchItem:
    return BatchItem(req, req.prompt_len - req.prefill_done, "prefill")


def one_decode(req: Request) -> BatchItem:
    return BatchItem(req, 1, "decode")


def run_simulation(
    trace_path: Path,
    policy: str,
    scheduler: Scheduler,
    output_path: Path,
    params: ModelParams,
    admit_request: AdmissionFn | None = None,
) -> dict:
    requests = load_trace(trace_path)
    pending = [(r.arrival_ms, r.request_id, r) for r in requests]
    heapq.heapify(pending)
    active: list[Request] = []
    now_ms = pending[0][0] if pending else 0.0
    steps = 0
    prefill_tokens = 0
    decode_tokens = 0
    total_step_time = 0.0

    while pending or any(not r.done for r in active):
        while pending and pending[0][0] <= now_ms:
            req = heapq.heappop(pending)[2]
            if admit_request is not None and not admit_request(active, now_ms, req, params):
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
                actual_tpots = [
                    (req.output_times[i] - req.output_times[0]) / i
                    for i in range(1, len(req.output_times))
                ]
                deadline_tpots = [
                    max(0.0, (req.output_times[i] - req.arrival_ms - req.ttft_slo_ms) / i)
                    for i in range(1, len(req.output_times))
                ]
                max_tpot = max(actual_tpots)
                slo_tpot = max(deadline_tpots)
            else:
                max_tpot = 0.0
                slo_tpot = 0.0
            slo_met = ttft <= req.ttft_slo_ms and slo_tpot <= req.tpot_slo_ms
        else:
            ttft = 2.0 * req.ttft_slo_ms
            max_tpot = 2.0 * req.tpot_slo_ms
            slo_tpot = max_tpot
            slo_met = False
        rows.append(
            {
                "request_id": req.request_id,
                "arrival_ms": req.arrival_ms,
                "prompt_len": req.prompt_len,
                "output_len": req.output_len,
                "ttft_ms": ttft,
                "max_tpot_ms": max_tpot,
                "max_envelope_tpot_ms": slo_tpot,
                "ttft_slo_ms": req.ttft_slo_ms,
                "tpot_slo_ms": req.tpot_slo_ms,
                "slo_met": slo_met,
            }
        )

    arrival_span_s = max(1e-9, (requests[-1].arrival_ms - requests[0].arrival_ms) / 1000.0)
    success_fraction = sum(1 for r in rows if r["slo_met"]) / len(rows)
    summary = {
        "policy": policy,
        "trace": str(trace_path),
        "requests": len(requests),
        "offered_rps": len(requests) / arrival_span_s,
        "steps": steps,
        "prefill_tokens": prefill_tokens,
        "decode_tokens": decode_tokens,
        "mean_step_ms": total_step_time / max(steps, 1),
        "effective_rps": len(requests) / arrival_span_s * success_fraction,
        "slo_violation_rate": 1.0 - success_fraction,
        "ttft_p95_ms": percentile([r["ttft_ms"] for r in rows], 95),
        "ttft_p99_ms": percentile([r["ttft_ms"] for r in rows], 99),
        "tpot_p95_ms": percentile([r["max_tpot_ms"] for r in rows], 95),
        "tpot_p99_ms": percentile([r["max_tpot_ms"] for r in rows], 99),
        "envelope_tpot_p95_ms": percentile([r["max_envelope_tpot_ms"] for r in rows], 95),
        "envelope_tpot_p99_ms": percentile([r["max_envelope_tpot_ms"] for r in rows], 99),
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
