"""Evaluator-side LLM serving simulator (held-out test copy).

Self-contained: this file does not import anything from data/, method/ or
baseline/, so nothing outside eval/test/ can change the workload, the hardware
model or the comparison baselines used for the held-out score.

The evaluator owns everything that a scheduler is compared on:

* the workload: request arrivals and prompt/output lengths sampled from the
  QwenTrace head segment (an evaluator-owned copy, ``eval/test/qwen_trace_head.csv``) and replayed at a
  chosen request rate;
* the hardware: a step-time model for Qwen3-14B on one H20-class GPU
  (``HARDWARE``).  Schedulers only receive an offline linear fit of it
  (``PROFILE``) plus the measured duration of each executed step;
* the clock, the token bookkeeping and every latency number.

A scheduler is driven step by step through the JSON-lines protocol in
``data/sched_client.py``.  It sees only requests that have already arrived and
never their output lengths.  Every reply is validated; an invalid reply, a
crash or a timeout fails the run.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import queue
import random
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

# --------------------------------------------------------------------------
# Hardware model (true step time) and the published linear profile.
# --------------------------------------------------------------------------

# Qwen3-14B on one H20-96GB, the configuration of the paper's Table 4.
#   a  fixed per-step cost (streaming ~30 GB of weights at 4.0 TB/s plus engine
#      and launch overhead);
#   b  per new token (dense GEMMs).  A pure bf16 spec estimate gives
#      ~0.25 ms/token, which saturates the node at about half the load the
#      paper's Figure 5(f) shows (there every system still meets the SLO for
#      ~all requests at 1 RPS and FairBatching-PAB serves 3.2 RPS within SLO
#      at 4 RPS offered).  b = 0.15 ms/token (~6.7k prefill tokens/s) puts the
#      simulated node in that same regime;
#   c  per token of context of each request in the batch (reading ~160 KB of
#      KV cache per token at 4.0 TB/s);
#   d  per (query token x key token) pair of prefill attention, which the
#      linear profile cannot express (long prompts, chunked prefill);
#   noise multiplicative per-step jitter (uniform +-1%).
HARDWARE = {
    "a": 8.0,
    "b": 0.15,
    "c": 4.0e-5,
    "d": 1.5e-6,
    "noise": 0.01,
    "max_batch_tokens": 8192,
}

# Least-squares fit of ``a + b*new_tokens + c*total_context`` to HARDWARE on
# random mixed batches (data/profile_hardware.py): ~0.8% median / ~2.8% p95
# relative error, comparable to the +-1.3% the paper reports for its linear
# model.  This is all a scheduler is told about the hardware.
PROFILE = {"a": 7.70, "b": 0.1515, "c": 3.97e-05, "max_batch_tokens": 8192}

HORIZON_MS = 600_000.0  # simulated time allowed after the last arrival
IDLE_MS = 5.0  # clock advance when a scheduler returns an empty batch
MAX_ITERATIONS = 2_000_000
# Validity limits for scheduler programs (not metrics; simulated time never
# depends on how long a scheduler thinks).  CPU limits use the scheduler
# process's own CPU time, so a busy or oversubscribed machine cannot trip them;
# the wall-clock limits only catch programs that stall (stop reading, sleep).
# The reference needs ~1-3 s of CPU per run.
STEP_TIMEOUT_S = 30.0  # wall-clock: one reply, or one write of a message
RUN_WALL_FLOOR_S = 600.0  # per-run wall-clock backstop: max(floor, factor x CPU used)
RUN_WALL_PER_CPU = 100.0
RUN_CPU_LIMIT_S = 300.0  # scheduler CPU seconds per run
CPU_CHECK_EVERY = 256  # messages between CPU-time checks


class RunFailure(Exception):
    """Raised when a scheduler crashes, times out or sends an invalid reply."""


# --------------------------------------------------------------------------
# Workload
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class TraceRequest:
    rid: int
    arrival_ms: float
    prompt_len: int
    output_len: int
    ttft_slo_ms: float
    tpot_slo_ms: float


def load_pool(path: Path) -> list[tuple[float, int, int]]:
    rows = []
    with Path(path).open() as f:
        for row in csv.DictReader(f):
            rows.append((float(row["timestamp_s"]), int(row["input_length"]), int(row["output_length"])))
    return rows


def derive_seed(*parts: object) -> int:
    payload = ":".join(str(p) for p in parts).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def make_instance(
    pool: list[tuple[float, int, int]],
    seed: int,
    n_requests: int,
    rps: float,
    ttft_slo_ms: float,
    tpot_slo_ms: float,
    neighborhood: int = 300,
) -> list[TraceRequest]:
    """Sample one workload instance.

    Arrivals are a window of ``n_requests`` consecutive real arrivals, rescaled
    so the offered rate is exactly ``rps`` (``n_requests`` arrivals over the
    window span).  Prompt/output length pairs are bootstrapped from the
    requests around that window, so the instance is not a verbatim slice of
    the public trace.  Request ids are random and carry no ordering.
    """
    rng = random.Random(seed)
    if n_requests < 2 or n_requests > len(pool):
        raise ValueError("bad n_requests")
    start = rng.randrange(0, len(pool) - n_requests + 1)
    times = [pool[start + i][0] for i in range(n_requests)]
    span = times[-1] - times[0]
    if span <= 0:
        raise ValueError("degenerate window")
    scale_ms = (n_requests / rps) * 1000.0 / span
    lo = max(0, start - neighborhood)
    hi = min(len(pool), start + n_requests + neighborhood)
    ids = rng.sample(range(100_000, 1_000_000), n_requests)
    out = []
    for i in range(n_requests):
        _, prompt, output = pool[rng.randrange(lo, hi)]
        out.append(
            TraceRequest(
                rid=ids[i],
                arrival_ms=round((times[i] - times[0]) * scale_ms, 3),
                prompt_len=max(1, prompt),
                output_len=max(1, output),
                ttft_slo_ms=float(ttft_slo_ms),
                tpot_slo_ms=float(tpot_slo_ms),
            )
        )
    return out


# --------------------------------------------------------------------------
# Simulation
# --------------------------------------------------------------------------


class _State:
    __slots__ = (
        "req",
        "prefill_done",
        "generated",
        "status",
        "t0",
        "ref",
        "max_raw",
        "max_tpot",
        "env_ok",
    )

    def __init__(self, req: TraceRequest):
        self.req = req
        self.prefill_done = 0
        self.generated = 0
        self.status = "pending"  # pending | active | rejected | served
        self.t0 = None
        self.ref = None
        self.max_raw = 0.0
        self.max_tpot = 0.0
        self.env_ok = True


def _row(st: _State) -> list:
    r = st.req
    return [r.rid, r.arrival_ms, r.prompt_len, st.prefill_done, st.generated, r.ttft_slo_ms, r.tpot_slo_ms]


def _as_int(value) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        if isinstance(value, float) and value.is_integer():
            return int(value)
        raise RunFailure(f"expected integer, got {value!r}")
    return value


def simulate(
    requests: list[TraceRequest],
    policy,
    noise_seed: int,
    hardware: dict = HARDWARE,
    profile: dict = PROFILE,
) -> list[dict]:
    """Drive ``policy`` over ``requests``; return one outcome record per request."""
    states = {r.rid: _State(r) for r in requests}
    order = sorted(requests, key=lambda r: (r.arrival_ms, r.rid))
    rng = random.Random(noise_seed)
    a, b, c, d = hardware["a"], hardware["b"], hardware["c"], hardware["d"]
    noise = hardware["noise"]
    max_tokens = int(hardware["max_batch_tokens"])
    horizon = order[-1].arrival_ms + HORIZON_MS

    policy.start({"type": "init", "profile": dict(profile)})
    try:
        active: dict[int, _State] = {}
        arrived: list[int] = []
        idx = 0
        now = order[0].arrival_ms
        last_step = None
        iterations = 0
        while True:
            while idx < len(order) and order[idx].arrival_ms <= now:
                st = states[order[idx].rid]
                st.status = "active"
                active[st.req.rid] = st
                arrived.append(st.req.rid)
                idx += 1
            if not active:
                if idx >= len(order):
                    break
                now = order[idx].arrival_ms
                last_step = None
                continue
            iterations += 1
            if now > horizon or iterations > MAX_ITERATIONS:
                break
            reply = policy.step(
                {
                    "type": "step",
                    "now": now,
                    "last_step_ms": last_step,
                    "arrived": arrived,
                    "requests": [_row(st) for st in active.values()],
                }
            )
            arrived = []
            if not isinstance(reply, dict):
                raise RunFailure("reply is not an object")
            for rid in reply.get("reject", []) or []:
                rid = _as_int(rid)
                st = active.get(rid)
                if st is None:
                    raise RunFailure(f"reject of unknown or inactive request {rid}")
                if st.prefill_done > 0:
                    raise RunFailure(f"reject of request {rid} whose prefill already started")
                st.status = "rejected"
                del active[rid]
            batch_raw = reply.get("batch", [])
            if not isinstance(batch_raw, list):
                raise RunFailure("batch is not a list")
            batch: list[tuple[_State, int, bool]] = []
            seen = set()
            total_tokens = 0
            for item in batch_raw:
                if not isinstance(item, (list, tuple)) or len(item) != 2:
                    raise RunFailure(f"malformed batch item {item!r}")
                rid, n = _as_int(item[0]), _as_int(item[1])
                st = active.get(rid)
                if st is None:
                    raise RunFailure(f"batch names unknown or inactive request {rid}")
                if rid in seen:
                    raise RunFailure(f"request {rid} appears twice in one batch")
                seen.add(rid)
                is_prefill = st.prefill_done < st.req.prompt_len
                if is_prefill:
                    if n < 1 or n > st.req.prompt_len - st.prefill_done:
                        raise RunFailure(f"invalid prefill chunk {n} for request {rid}")
                elif n != 1:
                    raise RunFailure(f"decode request {rid} must get exactly one token, got {n}")
                total_tokens += n
                batch.append((st, n, is_prefill))
            if total_tokens > max_tokens:
                raise RunFailure(f"batch has {total_tokens} tokens > {max_tokens}")
            if not batch:
                nxt = order[idx].arrival_ms if idx < len(order) else math.inf
                now = min(now + IDLE_MS, nxt) if nxt > now else now + IDLE_MS
                last_step = None
                continue
            ctx = 0
            attn = 0.0
            for st, n, is_prefill in batch:
                ctx += st.prefill_done + st.generated
                if is_prefill:
                    attn += n * (st.prefill_done + n / 2.0)
            eps = rng.uniform(-noise, noise)
            duration = (a + b * total_tokens + c * ctx + d * attn) * (1.0 + eps)
            end = now + duration
            for st, n, is_prefill in batch:
                r = st.req
                if is_prefill:
                    st.prefill_done += n
                    if st.prefill_done >= r.prompt_len:
                        st.generated = 1
                        st.t0 = end
                        st.ref = max(end, r.arrival_ms + r.ttft_slo_ms)
                        if end > r.arrival_ms + r.ttft_slo_ms:
                            st.env_ok = False
                else:
                    j = st.generated
                    st.generated += 1
                    st.max_raw = max(st.max_raw, (end - st.t0) / j)
                    st.max_tpot = max(st.max_tpot, (end - st.ref) / j)
                    if end > r.arrival_ms + r.ttft_slo_ms + r.tpot_slo_ms * j:
                        st.env_ok = False
                if st.generated >= r.output_len and st.prefill_done >= r.prompt_len:
                    st.status = "served"
                    del active[r.rid]
            now = end
            last_step = duration
    finally:
        policy.close()

    records = []
    for r in requests:
        st = states[r.rid]
        served = st.status == "served"
        records.append(
            {
                "status": st.status if st.status != "active" else "unfinished",
                "ttft_ms": (st.t0 - r.arrival_ms) if st.t0 is not None else None,
                "tpot_ms": st.max_tpot if served else None,
                "raw_tpot_ms": st.max_raw if served else None,
                "success": bool(served and st.env_ok),
            }
        )
    return records


# --------------------------------------------------------------------------
# Protocol objects (copy of the public client helper)
# --------------------------------------------------------------------------

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


# --------------------------------------------------------------------------
# Policy drivers
# --------------------------------------------------------------------------


class InProcessPolicy:
    """Drive an evaluator-owned policy object directly."""

    def __init__(self, factory):
        self.factory = factory
        self.policy = None

    def start(self, init_msg: dict) -> None:
        self._handle = handle_message
        self.policy = self.factory()
        self._handle(self.policy, init_msg)

    def step(self, msg: dict) -> dict:
        return json.loads(json.dumps(self._handle(self.policy, msg)))

    def close(self) -> None:
        self.policy = None


class SubprocessPolicy:
    """Drive a scheduler program over stdin/stdout JSON lines."""

    def __init__(self, command: list[str], cwd: Path, log_path: Path | None = None):
        if command and command[0] in ("python", "python3"):
            command = [sys.executable] + list(command[1:])
        self.command = command
        self.cwd = cwd
        self.log_path = log_path
        self.proc = None
        self.started_at = None
        self.cpu_used = 0.0
        self.messages = 0

    def start(self, init_msg: dict) -> None:
        log = open(self.log_path, "w") if self.log_path else subprocess.DEVNULL
        try:
            self.proc = subprocess.Popen(
                self.command,
                cwd=str(self.cwd),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=log,
                text=True,
                bufsize=1,
            )
        except OSError as exc:
            raise RunFailure(f"cannot start scheduler: {exc}") from exc
        finally:
            if log is not subprocess.DEVNULL:
                log.close()
        self.lines: queue.Queue = queue.Queue()
        self.outbox: queue.Queue = queue.Queue()
        self.written = threading.Event()
        self.write_error: list = []
        threading.Thread(target=self._reader, daemon=True).start()
        threading.Thread(target=self._writer, daemon=True).start()
        self.started_at = time.monotonic()
        self._send(init_msg)

    def _reader(self) -> None:
        assert self.proc is not None and self.proc.stdout is not None
        try:
            for line in self.proc.stdout:
                self.lines.put(line)
        except (OSError, ValueError):
            pass
        self.lines.put(None)

    def _writer(self) -> None:
        while True:
            payload = self.outbox.get()
            if payload is None:
                return
            try:
                self.proc.stdin.write(payload)
                self.proc.stdin.flush()
            except (BrokenPipeError, OSError, ValueError) as exc:
                self.write_error.append(exc)
            self.written.set()

    def _cpu_seconds(self) -> float | None:
        """CPU time (user + system) consumed so far by the scheduler process."""
        try:
            with open(f"/proc/{self.proc.pid}/stat") as f:
                fields = f.read().rsplit(")", 1)[1].split()
            return (int(fields[11]) + int(fields[12])) / os.sysconf("SC_CLK_TCK")
        except (OSError, ValueError, IndexError, AttributeError):
            return None

    def _check_limits(self) -> None:
        cpu = self._cpu_seconds()
        if cpu is not None:
            self.cpu_used = cpu
            if cpu > RUN_CPU_LIMIT_S:
                raise RunFailure("scheduler exceeded its CPU limit for one run")
        # A stalling program (sleeping, waiting) uses wall time but no CPU; an
        # honest one on a busy machine uses both.
        limit = max(RUN_WALL_FLOOR_S, RUN_WALL_PER_CPU * (cpu or 0.0))
        if time.monotonic() - self.started_at > limit:
            raise RunFailure("run exceeded its wall-clock backstop (scheduler stalled)")

    def _send(self, msg: dict) -> dict:
        self.messages += 1
        if self.messages % CPU_CHECK_EVERY == 0 or time.monotonic() - self.started_at > RUN_WALL_FLOOR_S:
            self._check_limits()
        self.written.clear()
        self.outbox.put(json.dumps(msg, separators=(",", ":")) + "\n")
        if not self.written.wait(STEP_TIMEOUT_S):
            raise RunFailure("scheduler stopped reading its input")
        if self.write_error:
            raise RunFailure(f"scheduler pipe closed: {self.write_error[0]}")
        while True:
            try:
                line = self.lines.get(timeout=STEP_TIMEOUT_S)
            except queue.Empty as exc:
                raise RunFailure("scheduler reply timed out") from exc
            if line is None:
                raise RunFailure("scheduler exited before replying")
            line = line.strip()
            if not line:
                continue
            try:
                return json.loads(line)
            except json.JSONDecodeError as exc:
                raise RunFailure(f"invalid JSON from scheduler: {line[:120]!r}") from exc

    def step(self, msg: dict) -> dict:
        return self._send(msg)

    def close(self) -> None:
        if self.proc is None:
            return
        cpu = self._cpu_seconds()
        if cpu is not None:
            self.cpu_used = cpu
        if self.proc.poll() is None:
            self.written.clear()
            self.outbox.put(json.dumps({"type": "end"}) + "\n")
            self.written.wait(2.0)
        self.outbox.put(None)
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()
        if cpu is None:  # no /proc: fall back to wall-clock time
            self.cpu_used = time.monotonic() - self.started_at


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------


def percentile(values: list[float], pct: float) -> float:
    values = sorted(values)
    if not values:
        raise ValueError("empty")
    idx = (len(values) - 1) * pct / 100.0
    lo = int(idx)
    hi = min(lo + 1, len(values) - 1)
    frac = idx - lo
    return values[lo] * (1 - frac) + values[hi] * frac


def success_fraction(records: list[dict]) -> float:
    return sum(1 for r in records if r["success"]) / len(records)


def censored(records: list[dict], key: str, cap: float) -> list[float]:
    """Per-request latency where any request that was not fully served counts
    at ``cap``; served values are clipped to ``cap``."""
    out = []
    for r in records:
        v = r[key] if r["status"] == "served" else None
        out.append(cap if v is None else min(cap, max(0.0, float(v))))
    return out


# --------------------------------------------------------------------------
# Evaluator-owned baselines (comparison point for goodput gains and the
# latency attainment floor).  Identical logic to baseline/schedulers.py at the
# time the test was frozen.
# --------------------------------------------------------------------------

SARATHI_BUDGET = {50.0: 256, 100.0: 512}


class FrozenVllmVanilla(Policy):
    def select(self, active, now_ms):
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


class FrozenVllmSarathi(Policy):
    def select(self, active, now_ms):
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


FROZEN_BASELINES = {"vllm_vanilla": FrozenVllmVanilla, "vllm_sarathi": FrozenVllmSarathi}
