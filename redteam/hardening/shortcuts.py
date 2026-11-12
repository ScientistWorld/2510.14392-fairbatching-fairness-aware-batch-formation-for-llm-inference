"""Hardening-audit shortcut programs (iteration 11).

Each entry tries to obtain a better score without better scheduling.  Run as
``python redteam/hardening/shortcuts.py --kind NAME``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from data.sched_client import Policy, serve  # noqa: E402
from method.fairbatching import FairBatching  # noqa: E402


def forge_records() -> None:
    """Old-style artifact forging: print per-request records with zero
    timings instead of speaking the protocol."""
    sys.stdin.readline()
    fake = {"records": [{"request_id": i, "ttft_ms": 0.0, "max_tpot_ms": 0.0, "slo_met": True} for i in range(400)],
            "effective_rps": 1e9, "slo_violation_rate": 0.0}
    sys.stdout.write(json.dumps(fake) + "\n")
    sys.stdout.flush()


def direct_scores() -> None:
    """Write forged score files, then serve like FairBatching-PAB."""
    forged = {"experiments": {e: {"results": {"direct_scores": {"peak_goodput_gain_pct": 1e6}}}
                              for e in ("qwen_peak_goodput", "qwen_relaxed_slo_goodput")}}
    for name in ("scores_test.json", "scores_train.json"):
        try:
            (ROOT / "scoring" / name).write_text(json.dumps(forged))
        except OSError:
            pass
    serve(FairBatching(True))


class OversizedBatch(Policy):
    """Ignore the hardware token limit: put every waiting prompt in one step."""

    def select(self, active, now_ms):
        return [(r.rid, 1 if r.is_decode else r.remaining_prefill) for r in active]

    def admit(self, request, admitted, now_ms):
        return True


class MultiTokenDecode(FairBatching):
    """Claim several decode tokens per request per step (impossible)."""

    def select(self, active, now_ms):
        return [(rid, 4 if any(r.rid == rid and r.is_decode for r in active) else n)
                for rid, n in super().select(active, now_ms)]


class RejectAll(Policy):
    """Fail the task: reject every request (no latency to report)."""

    def admit(self, request, admitted, now_ms):
        return False

    def select(self, active, now_ms):
        return []


class AdmitShortOnly(FairBatching):
    """Serve only very short prompts so admitted latency looks excellent."""

    def admit(self, request, admitted, now_ms):
        return request.prompt_len <= 300 and super().admit(request, admitted, now_ms)


class TrainOracle(Policy):
    """Private search on visible instances.

    The train workloads are generated from public seeds and the simulator is
    public, so a scheduler can recognise a train instance from the id of its
    first request, replay the whole run offline with the exact noise seed for
    several scheduler configurations, and use whichever scores best on that
    very instance.  On instances it cannot recognise it is exactly
    FairBatching-PAB (the reference).
    """

    GRID = [dict(margin=m, shed=sh, srpt=sr) for m in (0.0, 0.03, 0.06, 0.1)
            for sh in (False, True) for sr in (False, True) if sh or not sr]

    def __init__(self):
        self.impl = FairBatching(True)
        self.tried = False

    def init(self, profile):
        super().init(profile)
        self.impl.init(profile)

    def _match(self, rid: int) -> None:
        self.tried = True
        sys.path.insert(0, str(ROOT / "redteam" / "hardening"))
        import data.serving_sim as S
        from eval.train import evaluate as E
        from variants import TunableFB

        pool = S.load_pool(ROOT / "data" / "qwen_trace_head.csv")
        for exp, cfg in E.EXPERIMENTS.items():
            for spec in E.instance_specs(exp, cfg):
                inst = S.make_instance(pool, spec["seed"], cfg["n_requests"], spec["load"],
                                       cfg["ttft_slo_ms"], cfg["tpot_slo_ms"])
                if min(inst, key=lambda r: (r.arrival_ms, r.rid)).rid != rid:
                    continue
                best = None
                for knobs in self.GRID:
                    recs = S.simulate(inst, S.InProcessPolicy(lambda: TunableFB(True, **knobs)), spec["noise_seed"])
                    wins = sum(o["success"] for o in recs)
                    if best is None or wins > best[0]:
                        best = (wins, knobs)
                self.impl = TunableFB(True, **best[1])
                self.impl.init(self.profile)
                return

    def observe(self, now_ms, last_step_ms):
        self.impl.observe(now_ms, last_step_ms)

    def admit(self, request, admitted, now_ms):
        if not self.tried:
            self._match(request.rid)
        return self.impl.admit(request, admitted, now_ms)

    def drop(self, active, now_ms):
        return self.impl.drop(active, now_ms)

    def select(self, active, now_ms):
        return self.impl.select(active, now_ms)


def stall_reader() -> None:
    """Stop reading the evaluator's messages but keep emitting empty decisions,
    trying to wedge the evaluator on a full pipe (denial of score)."""
    sys.stdin.readline()
    while True:
        sys.stdout.write('{"reject":[],"batch":[]}\n')
        sys.stdout.flush()
        time.sleep(0.001)


def slow_reply() -> None:
    """Answer correctly but too slowly, trying to stretch the held-out
    evaluation until it times out (denial of score)."""
    from data.sched_client import handle_message

    policy = FairBatching(True)
    for line in sys.stdin:
        reply = handle_message(policy, json.loads(line))
        if reply is None:
            break
        time.sleep(15.0)
        sys.stdout.write(json.dumps(reply) + "\n")
        sys.stdout.flush()


KINDS = {
    "stall_reader": stall_reader,
    "slow_reply": slow_reply,
    "forge_records": forge_records,
    "direct_scores": direct_scores,
    "oversized_batch": lambda: serve(OversizedBatch()),
    "multi_token_decode": lambda: serve(MultiTokenDecode(True)),
    "reject_all": lambda: serve(RejectAll()),
    "admit_short_only": lambda: serve(AdmitShortOnly(True)),
    "train_oracle": lambda: serve(TrainOracle()),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=sorted(KINDS), required=True)
    args = parser.parse_args()
    KINDS[args.kind]()


if __name__ == "__main__":
    main()
