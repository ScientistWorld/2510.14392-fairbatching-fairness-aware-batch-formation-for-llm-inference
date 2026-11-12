"""Client side of the scheduler protocol used by the evaluators.

A scheduler is an ordinary program that talks to the evaluator over
stdin/stdout, one JSON object per line.  The evaluator owns the request
trace, the hardware cost model and the clock; the scheduler only decides,
at every engine step, which requests to admit or reject and which tokens to
put in the next batch.  See ``briefing/evaluation.md`` for the full message
specification.

This module turns that loop into two callbacks so a scheduler only has to
implement a :class:`Policy` subclass and call :func:`serve`::

    class MyPolicy(Policy):
        def select(self, active, now_ms):
            return [(r.rid, 1 if r.is_decode else r.remaining_prefill) for r in active]

    if __name__ == "__main__":
        serve(MyPolicy())

Nothing here is specific to any scheduling method.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from typing import Iterable


@dataclass
class RequestView:
    """Read-only snapshot of one admitted, unfinished request."""

    rid: int
    arrival_ms: float
    prompt_len: int
    prefill_done: int
    generated: int
    ttft_slo_ms: float
    tpot_slo_ms: float

    @property
    def is_prefill(self) -> bool:
        return self.prefill_done < self.prompt_len

    @property
    def is_decode(self) -> bool:
        return self.prefill_done >= self.prompt_len

    @property
    def remaining_prefill(self) -> int:
        return max(0, self.prompt_len - self.prefill_done)

    @property
    def context_len(self) -> int:
        return self.prefill_done + self.generated

    @property
    def started(self) -> bool:
        return self.prefill_done > 0

    def token_deadline(self, index: int) -> float:
        """SLO deadline of the output token with 0-based ``index``."""
        return self.arrival_ms + self.ttft_slo_ms + self.tpot_slo_ms * index

    def next_deadline(self) -> float:
        """Deadline of the next output token this request will emit."""
        return self.token_deadline(self.generated)

    def slack(self, now_ms: float) -> float:
        return self.next_deadline() - now_ms


@dataclass(frozen=True)
class Profile:
    """Offline-profiled linear step-time model published by the evaluator.

    ``estimate = a + b * new_tokens + c * total_context``.  It is a fit, not
    the exact cost: true step durations differ (see ``last_step_ms``).
    """

    a: float
    b: float
    c: float
    max_batch_tokens: int


class Policy:
    """Base class for schedulers.  Override :meth:`select` (and optionally
    :meth:`admit`, :meth:`drop`, :meth:`observe`)."""

    profile: Profile

    def init(self, profile: Profile) -> None:
        self.profile = profile

    def observe(self, now_ms: float, last_step_ms: float | None) -> None:
        """Called once per step before admission with the measured duration
        of the previous executed batch (``None`` after an idle period)."""

    def admit(self, request: RequestView, admitted: list[RequestView], now_ms: float) -> bool:
        """Decide whether a newly arrived request is admitted.  ``admitted``
        holds every request already admitted (including earlier arrivals of
        this same step).  Rejected requests count as SLO failures."""
        return True

    def drop(self, active: list[RequestView], now_ms: float) -> Iterable[int]:
        """Optionally reject admitted requests whose prefill has not started."""
        return ()

    def select(self, active: list[RequestView], now_ms: float) -> list[tuple[int, int]]:
        """Return ``[(rid, new_tokens), ...]`` for the next batch."""
        raise NotImplementedError


def _views(rows: list[list]) -> list[RequestView]:
    return [
        RequestView(
            rid=int(r[0]),
            arrival_ms=float(r[1]),
            prompt_len=int(r[2]),
            prefill_done=int(r[3]),
            generated=int(r[4]),
            ttft_slo_ms=float(r[5]),
            tpot_slo_ms=float(r[6]),
        )
        for r in rows
    ]


def handle_message(policy: Policy, msg: dict) -> dict | None:
    """Process one evaluator message; return the reply (``None`` to exit)."""
    kind = msg.get("type")
    if kind == "init":
        p = msg["profile"]
        policy.init(Profile(float(p["a"]), float(p["b"]), float(p["c"]), int(p["max_batch_tokens"])))
        return {"ok": True}
    if kind == "end":
        return None
    if kind != "step":
        raise ValueError(f"unknown message type {kind!r}")
    now = float(msg["now"])
    policy.observe(now, msg.get("last_step_ms"))
    views = _views(msg["requests"])
    arrived = set(int(x) for x in msg.get("arrived", []))
    admitted = [v for v in views if v.rid not in arrived]
    new = sorted((v for v in views if v.rid in arrived), key=lambda v: (v.arrival_ms, v.rid))
    rejected: list[int] = []
    for view in new:
        if policy.admit(view, admitted, now):
            admitted.append(view)
        else:
            rejected.append(view.rid)
    dropped = [int(x) for x in policy.drop(admitted, now)]
    if dropped:
        dropped_set = set(dropped)
        admitted = [v for v in admitted if v.rid not in dropped_set]
        rejected.extend(dropped)
    batch = policy.select(admitted, now) if admitted else []
    return {"reject": rejected, "batch": [[int(rid), int(n)] for rid, n in batch]}


def serve(policy: Policy, stdin=None, stdout=None) -> None:
    """Run the JSON-lines protocol loop until the evaluator ends the run."""
    stdin = stdin or sys.stdin
    stdout = stdout or sys.stdout
    for line in stdin:
        line = line.strip()
        if not line:
            continue
        reply = handle_message(policy, json.loads(line))
        if reply is None:
            break
        stdout.write(json.dumps(reply, separators=(",", ":")) + "\n")
        stdout.flush()
