"""Evaluator-side LLM serving simulator (public copy used by eval/train).

The evaluator owns everything that a scheduler is compared on:

* the workload: request arrivals and prompt/output lengths sampled from the
  QwenTrace head segment (``data/qwen_trace_head.csv``) and replayed at a
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
import queue
import random
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

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
# Wall-clock validity limits for scheduler programs (not metrics): a reply must
# arrive within STEP_TIMEOUT_S and one run must finish within RUN_TIMEOUT_S.
# The reference needs ~1-2 s per run.
STEP_TIMEOUT_S = 10.0
RUN_TIMEOUT_S = 300.0


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
# Policy drivers
# --------------------------------------------------------------------------


class InProcessPolicy:
    """Drive a ``data.sched_client.Policy`` object directly (development aid)."""

    def __init__(self, factory):
        self.factory = factory
        self.policy = None

    def start(self, init_msg: dict) -> None:
        from data.sched_client import handle_message

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
        threading.Thread(target=self._reader, daemon=True).start()
        self.started_at = time.monotonic()
        self._send(init_msg)

    def _reader(self) -> None:
        assert self.proc is not None and self.proc.stdout is not None
        for line in self.proc.stdout:
            self.lines.put(line)
        self.lines.put(None)

    def _send(self, msg: dict) -> dict:
        if time.monotonic() - self.started_at > RUN_TIMEOUT_S:
            raise RunFailure("run exceeded its wall-clock limit")
        payload = json.dumps(msg, separators=(",", ":")) + "\n"
        error: list = []

        def write() -> None:
            try:
                self.proc.stdin.write(payload)
                self.proc.stdin.flush()
            except (BrokenPipeError, OSError, ValueError) as exc:
                error.append(exc)

        writer = threading.Thread(target=write, daemon=True)
        writer.start()
        writer.join(STEP_TIMEOUT_S)
        if writer.is_alive():
            raise RunFailure("scheduler stopped reading its input")
        if error:
            raise RunFailure(f"scheduler pipe closed: {error[0]}")
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
        if self.proc.poll() is None:
            closer = threading.Thread(target=self._send_end, daemon=True)
            closer.start()
            closer.join(2.0)
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()

    def _send_end(self) -> None:
        try:
            self.proc.stdin.write(json.dumps({"type": "end"}) + "\n")
            self.proc.stdin.flush()
            self.proc.stdin.close()
        except (BrokenPipeError, OSError, ValueError):
            pass


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
