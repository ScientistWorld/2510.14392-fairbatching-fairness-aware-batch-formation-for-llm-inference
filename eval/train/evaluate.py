"""Public train-split evaluator for scheduler programs.

Every scheduler registered under ``outputs/schedulers/<name>.json`` (a
manifest ``{"command": [...]}``) is run, through the JSON-lines protocol, on
public workload instances sampled from the QwenTrace head segment.  The
evaluator owns the trace, the hardware model, the clock and all latency
bookkeeping (``data/serving_sim.py``); a scheduler only returns admission and
batch decisions, which are validated every step.  Nothing a scheduler writes
is trusted as a number.

The held-out test evaluator runs the same experiments on instances drawn with
undisclosed seeds.  This file does not import from ``method/`` or
``eval/test/``.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import data.serving_sim as S  # noqa: E402
from eval.train.frozen_baselines import FROZEN_BASELINES  # noqa: E402

SPLIT = "train"
SEED_BASE = "fairbatching-train-v2"
POOL_PATH = ROOT / "data" / "qwen_trace_head.csv"

EXPERIMENTS = {
    "qwen_peak_goodput": {
        "kind": "sweep",
        "ttft_slo_ms": 500.0,
        "tpot_slo_ms": 50.0,
        "loads": [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
        "instances": 6,
        "n_requests": 400,
    },
    "qwen_latency_detail": {
        "kind": "latency",
        "ttft_slo_ms": 500.0,
        "tpot_slo_ms": 50.0,
        "load": 2.5,
        "instances": 12,
        "n_requests": 400,
    },
    "qwen_relaxed_slo_goodput": {
        "kind": "sweep",
        "ttft_slo_ms": 1000.0,
        "tpot_slo_ms": 100.0,
        "loads": [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0],
        "instances": 6,
        "n_requests": 400,
    },
}

POOL_SHA256 = "ac3aafb592e18949263a66fe74b20f664cd1aea8e9d3039a6746ff531c54bdbb"
_POOL = None


def workload_pool() -> list:
    """Load the request pool once, before any scheduler runs, and verify it."""
    global _POOL
    if _POOL is None:
        import hashlib

        data = POOL_PATH.read_bytes()
        if hashlib.sha256(data).hexdigest() != POOL_SHA256:
            raise SystemExit(f"{POOL_PATH} does not match the evaluator's workload data")
        _POOL = S.load_pool(POOL_PATH)
    return _POOL


# Total CPU time a scheduler program may spend over all its runs of one
# evaluation (the reference needs ~100-200 s).  Exceeding it fails the method.
METHOD_BUDGET_S = 3600.0

LATENCY_CAP_FACTOR = 20.0  # latency values are clipped at 20x the SLO

WORST = {
    "peak_goodput_gain_pct": -100.0,
    "mean_goodput_gain_pct": -100.0,
    "slo_attainment": 0.0,
    "ttft_p50_ms": LATENCY_CAP_FACTOR * 500.0,
    "ttft_p99_ms": LATENCY_CAP_FACTOR * 500.0,
    "tpot_p99_ms": LATENCY_CAP_FACTOR * 50.0,
}
METRICS = {
    "sweep": ["peak_goodput_gain_pct", "mean_goodput_gain_pct"],
    "latency": ["slo_attainment", "ttft_p50_ms", "ttft_p99_ms", "tpot_p99_ms"],
}


def instance_specs(name: str, cfg: dict) -> list[dict]:
    loads = cfg["loads"] if cfg["kind"] == "sweep" else [cfg["load"]]
    specs = []
    for load in loads:
        for k in range(cfg["instances"]):
            specs.append(
                {
                    "experiment": name,
                    "load": load,
                    "k": k,
                    # As in the paper, the same trace windows are replayed at
                    # every load of a sweep; only the time scale changes.
                    "seed": S.derive_seed(SEED_BASE, name, k, "workload"),
                    "noise_seed": S.derive_seed(SEED_BASE, name, load, k, "noise"),
                }
            )
    return specs


# The shipped schedulers are always scored; a missing manifest scores as a
# failed run instead of silently disappearing from the results.
EXPECTED_METHODS = ("fairbatching", "fairbatching_pab", "vllm_sarathi", "vllm_vanilla")


def discover_methods(outputs: Path) -> dict[str, object]:
    methods: dict[str, object] = {m: "manifest missing" for m in EXPECTED_METHODS}
    sched_dir = outputs / "schedulers"
    if not sched_dir.is_dir():
        return methods
    for path in sorted(sched_dir.glob("*.json")):
        if path.name.startswith("_"):
            continue
        try:
            manifest = json.loads(path.read_text())
            command = manifest["command"]
            if not isinstance(command, list) or not command or not all(isinstance(x, str) for x in command):
                raise ValueError("command must be a non-empty list of strings")
            methods[path.stem] = command
        except Exception as exc:  # an unreadable manifest is a failed method
            methods[path.stem] = f"invalid manifest: {exc}"
    return methods


def _run_task(task: tuple) -> tuple:
    method, command, spec, log_dir, spent, lock = task
    if spent.get(("failed", method)):
        # Already failed (scores worst everywhere): skip its remaining runs.
        return method, spec, None, "skipped after an earlier failure"
    if spent.get(method, 0.0) > METHOD_BUDGET_S:
        return method, spec, None, "RunFailure: scheduler exceeded its total CPU budget"
    result = _run_one(method, command, spec, log_dir)
    with lock:
        spent[method] = spent.get(method, 0.0) + result[4]
        if result[3] is not None:
            spent[("failed", method)] = True
    return result[:4]


def _run_one(method, command, spec, log_dir) -> tuple:
    cfg = EXPERIMENTS[spec["experiment"]]
    pool = workload_pool()
    inst = S.make_instance(
        pool, spec["seed"], cfg["n_requests"], spec["load"], cfg["ttft_slo_ms"], cfg["tpot_slo_ms"]
    )
    policy = None
    try:
        if isinstance(command, str):
            raise S.RunFailure(command)
        if command[0] == "__frozen__":
            policy = S.InProcessPolicy(FROZEN_BASELINES[command[1]])
        else:
            log_path = None
            if log_dir is not None:
                log_path = Path(log_dir) / f"{method}__{spec['experiment']}__{spec['load']}__{spec['k']}.log"
            policy = S.SubprocessPolicy(command, ROOT, log_path)
        records = S.simulate(inst, policy, spec["noise_seed"])
    except Exception as exc:  # any failure on the scheduler side fails the method
        return method, spec, None, f"{type(exc).__name__}: {exc}", getattr(policy, "cpu_used", 0.0)
    return method, spec, records, None, getattr(policy, "cpu_used", 0.0)


def available_cpus() -> int:
    """CPUs this process may actually use: affinity, capped by a cgroup quota."""
    try:
        n = len(os.sched_getaffinity(0))
    except AttributeError:
        n = os.cpu_count() or 1
    quota = None
    try:
        raw = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if raw[0] != "max":
            quota = float(raw[0]) / float(raw[1])
    except (OSError, ValueError, IndexError):
        try:
            q = float(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text())
            period = float(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text())
            if q > 0:
                quota = q / period
        except (OSError, ValueError):
            pass
    if quota is not None:
        n = min(n, max(1, int(quota)))
    for var in ("SLURM_CPUS_PER_TASK", "OMP_NUM_THREADS"):  # allocation hints
        try:
            n = min(n, int(os.environ[var]))
        except (KeyError, ValueError):
            pass
    return max(1, min(n, 16))


def summarize_sweep(cfg: dict, runs: dict) -> dict:
    per_load, per_instance = {}, {}
    for load in cfg["loads"]:
        effs = [load * S.success_fraction(runs[(load, k)]) for k in range(cfg["instances"])]
        per_load[load] = sum(effs) / len(effs)
        per_instance[str(load)] = effs
    return {
        "peak_effective_rps": max(per_load.values()),
        "mean_effective_rps": sum(per_load.values()) / len(per_load),
        "effective_rps_by_load": {str(k): v for k, v in per_load.items()},
        "effective_rps_by_instance": per_instance,
    }


def summarize_latency(cfg: dict, runs: dict) -> dict:
    records = [r for k in range(cfg["instances"]) for r in runs[(cfg["load"], k)]]
    attainment = S.success_fraction(records)
    by_instance = [S.success_fraction(runs[(cfg["load"], k)]) for k in range(cfg["instances"])]
    admitted = [r for r in records if r["status"] != "rejected"]
    ttft_cap = LATENCY_CAP_FACTOR * cfg["ttft_slo_ms"]
    tpot_cap = LATENCY_CAP_FACTOR * cfg["tpot_slo_ms"]
    out = {
        "slo_attainment": attainment,
        "rejected_fraction": 1.0 - len(admitted) / len(records),
        "slo_attainment_by_instance": by_instance,
    }
    if admitted:
        ttft = S.censored(admitted, "ttft_ms", ttft_cap)
        tpot = S.censored(admitted, "tpot_ms", tpot_cap)
        out.update(
            ttft_p50_ms=S.percentile(ttft, 50),
            ttft_p99_ms=S.percentile(ttft, 99),
            tpot_p99_ms=S.percentile(tpot, 99),
        )
    else:
        out.update(ttft_p50_ms=ttft_cap, ttft_p99_ms=ttft_cap, tpot_p99_ms=tpot_cap)
    return out


def worst(kind: str) -> dict:
    return {m: WORST[m] for m in METRICS[kind]}


def evaluate(outputs: Path, workers: int, log_dir: Path | None, only: list[str] | None = None) -> dict:
    if only:
        for name in list(EXPERIMENTS):
            if name not in only:
                del EXPERIMENTS[name]
    workload_pool()
    methods = discover_methods(outputs)
    frozen = {f"__baseline_{name}": ["__frozen__", name] for name in FROZEN_BASELINES}
    everyone = {**methods, **frozen}
    manager = multiprocessing.Manager()
    spent, lock = manager.dict(), manager.Lock()
    tasks = []
    for exp, cfg in EXPERIMENTS.items():
        for spec in instance_specs(exp, cfg):
            for method, command in everyone.items():
                tasks.append((method, command, spec, str(log_dir) if log_dir else None, spent, lock))
    runs: dict[str, dict] = {m: {} for m in everyone}
    failures: dict[str, str] = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for method, spec, records, error in pool.map(_run_task, tasks, chunksize=1):
            if error is not None:
                if method not in failures or failures[method].endswith("skipped after an earlier failure"):
                    failures[method] = f"{spec['experiment']} load={spec['load']} k={spec['k']}: {error}"
                continue
            runs[method].setdefault(spec["experiment"], {})[(spec["load"], spec["k"])] = records
    frozen_failed = [m for m in frozen if m in failures]
    if frozen_failed:
        raise SystemExit(f"evaluator baselines failed: {[failures[m] for m in frozen_failed]}")

    experiments = {}
    for exp, cfg in EXPERIMENTS.items():
        summaries = {}
        for method in everyone:
            if method in failures:
                continue
            fn = summarize_sweep if cfg["kind"] == "sweep" else summarize_latency
            summaries[method] = fn(cfg, runs[method][exp])
        base = [summaries[m] for m in frozen]
        results, details = {}, {}
        for method in methods:
            if method in failures:
                results[method] = worst(cfg["kind"])
                details[method] = {"failed": failures[method]}
                continue
            s = summaries[method]
            if cfg["kind"] == "sweep":
                best_peak = max(b["peak_effective_rps"] for b in base)
                best_mean = max(b["mean_effective_rps"] for b in base)
                results[method] = {
                    "peak_goodput_gain_pct": (s["peak_effective_rps"] / best_peak - 1.0) * 100.0,
                    "mean_goodput_gain_pct": (s["mean_effective_rps"] / best_mean - 1.0) * 100.0,
                }
            else:
                # A method that keeps fewer requests within SLO than the
                # weakest baseline has failed the task: its latency of the
                # few requests it did serve does not count.
                floor = min(b["slo_attainment"] for b in base)
                if s["slo_attainment"] < floor:
                    results[method] = worst("latency")
                    s = dict(s, gate=f"slo_attainment {s['slo_attainment']:.4f} below baseline floor {floor:.4f}")
                else:
                    results[method] = {m: s[m] for m in METRICS["latency"]}
            details[method] = s
        experiments[exp] = {
            "split": SPLIT,
            "results": results,
            "details": details,
            "evaluator_baselines": {m.replace("__baseline_", ""): summaries[m] for m in frozen},
        }
    return {
        "_meta": {
            "split": SPLIT,
            "failed_methods": failures and {m: e for m, e in failures.items() if m in methods},
        },
        "experiments": experiments,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs", type=Path, default=ROOT / "outputs")
    parser.add_argument("--scores", type=Path, default=ROOT / "scoring" / "scores_train.json")
    parser.add_argument("--workers", type=int, default=available_cpus())
    parser.add_argument("--log-dir", type=Path, default=None, help="write scheduler stderr here")
    parser.add_argument("--experiments", default=None, help="comma-separated subset (development aid)")
    args = parser.parse_args()
    if args.log_dir:
        args.log_dir.mkdir(parents=True, exist_ok=True)
    only = args.experiments.split(",") if args.experiments else None
    scores = evaluate(args.outputs, args.workers, args.log_dir, only)
    args.scores.parent.mkdir(parents=True, exist_ok=True)
    args.scores.write_text(json.dumps(scores, indent=2))
    summary = {e: v["results"] for e, v in scores["experiments"].items()}
    print(json.dumps(summary, indent=2))
    if scores["_meta"]["failed_methods"]:
        print("failed methods:", json.dumps(scores["_meta"]["failed_methods"], indent=2), file=sys.stderr)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        sys.exit(1)
