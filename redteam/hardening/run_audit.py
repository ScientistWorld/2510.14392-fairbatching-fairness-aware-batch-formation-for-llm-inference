"""Hardening audit (iteration 11): run variants and shortcuts through the gym.

Every attempt is registered as a scheduler manifest and scored by the real
evaluators (``eval/train/evaluate.py`` and ``eval/test/evaluate.py``, the
latter exactly as ``scripts/evaluate_test.sh`` runs it, only pointed at the
audit's own outputs directory).  Results are written under
``redteam/hardening/runs/``; every step is skipped when its result file
already exists, so the audit resumes after interruption.

Usage: ``python redteam/hardening/run_audit.py``
"""

from __future__ import annotations

import json
import os
import random
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AUD = ROOT / "redteam" / "hardening"
RUNS = AUD / "runs"
PY = sys.executable

V = ["python", "redteam/hardening/variants.py", "--variant"]
SC = ["python", "redteam/hardening/shortcuts.py", "--kind"]

REFERENCE_SET = {
    "fairbatching": ["python", "method/fairbatching.py", "--variant", "vanilla"],
    "fairbatching_pab": ["python", "method/fairbatching.py", "--variant", "pab"],
    "vllm_vanilla": ["python", "baseline/schedulers.py", "--policy", "vllm_vanilla"],
    "vllm_sarathi": ["python", "baseline/schedulers.py", "--policy", "vllm_sarathi"],
}
VARIANT_SET = {
    "better_margin_pab": V + ["margin_pab"],
    "better_shed_pab": V + ["shed_pab"],
    "better_srpt_shed_pab": V + ["srpt_shed_pab"],
    "better_exact_cost_pab": V + ["exact_cost_pab"],
    "weaker_fixed_batch_fb": V + ["fixed_batch_fb"],
    "weaker_fixed_batch_pab": V + ["fixed_batch_pab"],
    "weaker_serial_fifo": V + ["serial_fifo"],
}
SHORTCUT_SET = {
    "shortcut_forge_records": SC + ["forge_records"],
    "shortcut_direct_scores": SC + ["direct_scores"],
    "shortcut_oversized_batch": SC + ["oversized_batch"],
    "shortcut_multi_token_decode": SC + ["multi_token_decode"],
    "shortcut_reject_all": SC + ["reject_all"],
    "shortcut_admit_short_only": SC + ["admit_short_only"],
    "shortcut_train_oracle": SC + ["train_oracle"],
    "shortcut_crash": ["python", "-c", "import sys; sys.exit(3)"],
    "shortcut_stall_reader": SC + ["stall_reader"],
    "shortcut_slow_reply": SC + ["slow_reply"],
}
RAW_MANIFESTS = {"shortcut_bad_manifest": "{not json"}
STATIC_BUDGETS = [128, 256, 512, 1024, 8192]
# Attempts scored on the public train split as well (to compare train and
# held-out behaviour); everything else runs on the held-out split only.
TRAIN_AUDIT = {"fairbatching_pab", "better_margin_pab", "better_shed_pab", "shortcut_train_oracle",
               "bruteforce_static_budget"}
OFFICIAL = ["scores.json", "scores_train.json", "scores_test.json"]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def write_outputs(path: Path, commands: dict, raw: dict | None = None) -> Path:
    sched = path / "schedulers"
    if path.exists():
        shutil.rmtree(path)
    sched.mkdir(parents=True)
    for name, cmd in commands.items():
        (sched / f"{name}.json").write_text(json.dumps({"command": cmd}))
    for name, text in (raw or {}).items():
        (sched / f"{name}.json").write_text(text)
    return path


def run_eval(evaluator: str, outputs: Path, scores: Path, extra: list[str] | None = None, cwd: Path = ROOT) -> dict:
    if scores.exists():
        log(f"skip (exists): {scores.relative_to(ROOT)}")
        return json.loads(scores.read_text())
    t = time.time()
    cmd = [PY, evaluator, "--outputs", str(outputs), "--scores", str(scores)] + (extra or [])
    log("run: " + " ".join(cmd[1:]))
    proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
    (scores.parent / (scores.stem + ".log")).write_text(proc.stdout[-20000:] + "\n--- stderr ---\n" + proc.stderr[-20000:])
    if proc.returncode != 0:
        raise SystemExit(f"evaluator failed ({proc.returncode}): {proc.stderr[-3000:]}")
    log(f"done in {time.time() - t:.0f}s: {scores.relative_to(ROOT)}")
    return json.loads(scores.read_text())


def results(scores: dict) -> dict:
    return {e: v["results"] for e, v in scores["experiments"].items()}


def rank_table(test_scores: dict, reference: dict, ref_vector: dict, cohort: list[str]) -> dict:
    res = results(test_scores)
    out = {m: 0.0 for m in res[next(iter(res))]}
    stds = {}
    for exp, spec in reference["experiments"].items():
        w = spec["weight"]
        for metric, mdef in spec["metrics"].items():
            vals = [res[exp][m][metric] for m in cohort if m in res[exp]]
            sd = statistics.pstdev(vals) if len(vals) > 1 else 0.0
            stds[f"{exp}.{metric}"] = sd
            if sd == 0:
                continue
            direction = 1.0 if mdef["higher_is_better"] else -1.0
            for m in out:
                out[m] += w * abs(mdef["coefficient"]) * direction * (res[exp][m][metric] - ref_vector[exp][metric]) / sd
    return {"ranks": out, "stds": stds}


def bootstrap_paired(test_scores: dict, a: str, b: str, n_boot: int = 2000, seed: int = 0) -> dict:
    """Paired bootstrap over workload instances of metric(a) - metric(b)."""
    rng = random.Random(seed)
    out = {}
    for exp, data in test_scores["experiments"].items():
        da, db = data["details"].get(a, {}), data["details"].get(b, {})
        if "effective_rps_by_instance" in da and "effective_rps_by_instance" in db:
            base = data["evaluator_baselines"]
            loads = list(da["effective_rps_by_instance"])
            k = len(da["effective_rps_by_instance"][loads[0]])

            def stat(idx):
                def peak_mean(d):
                    per = [statistics.mean(d["effective_rps_by_instance"][L][i] for i in idx[L]) for L in loads]
                    return max(per), statistics.mean(per)
                pa, ma = peak_mean(da)
                pb, mb = peak_mean(db)
                bp = max(peak_mean(x)[0] for x in base.values())
                bm = max(peak_mean(x)[1] for x in base.values())
                return (pa - pb) / bp * 100.0, (ma - mb) / bm * 100.0

            full = stat({L: list(range(k)) for L in loads})
            draws = [stat({L: [rng.randrange(k) for _ in range(k)] for L in loads}) for _ in range(n_boot)]
            for j, metric in enumerate(["peak_goodput_gain_pct", "mean_goodput_gain_pct"]):
                vals = sorted(d[j] for d in draws)
                out[f"{exp}.{metric}"] = {
                    "diff": full[j],
                    "ci95": [vals[int(0.025 * n_boot)], vals[int(0.975 * n_boot)]],
                    "boot_sd": statistics.pstdev(vals),
                }
        elif "slo_attainment_by_instance" in da and "slo_attainment_by_instance" in db:
            xa, xb = da["slo_attainment_by_instance"], db["slo_attainment_by_instance"]
            k = len(xa)
            diffs = [xa[i] - xb[i] for i in range(k)]
            draws = sorted(statistics.mean(diffs[rng.randrange(k)] for _ in range(k)) for _ in range(n_boot))
            out[f"{exp}.slo_attainment"] = {
                "diff": statistics.mean(diffs),
                "ci95": [draws[int(0.025 * n_boot)], draws[int(0.975 * n_boot)]],
                "boot_sd": statistics.pstdev(draws),
            }
    return out


def main() -> None:
    # Shortcut programs may write into scoring/; keep the official files intact.
    snapshot = {n: (ROOT / "scoring" / n).read_bytes() for n in OFFICIAL if (ROOT / "scoring" / n).exists()}
    try:
        audit_main()
    finally:
        forged = {n: (ROOT / "scoring" / n).read_bytes() != b for n, b in snapshot.items()}
        for n, b in snapshot.items():
            (ROOT / "scoring" / n).write_bytes(b)
        log(f"official score files restored (modified by an attempt during the audit: {forged})")


def audit_main() -> None:
    RUNS.mkdir(parents=True, exist_ok=True)
    log(f"os.cpu_count: {os.cpu_count()}; affinity: {len(os.sched_getaffinity(0))}")
    reference = json.loads((ROOT / "scoring" / "reference.json").read_text())

    # 1. Static-budget brute force: sweep on public train, select per experiment.
    static_cmds = {f"static_budget_{b}": V + ["static_budget", "--budgets", f"50:{b},100:{b}"] for b in STATIC_BUDGETS}
    out_static = write_outputs(RUNS / "outputs_static", static_cmds)
    st_train = run_eval("eval/train/evaluate.py", out_static, RUNS / "scores_train_static.json",
                        ["--experiments", "qwen_peak_goodput,qwen_relaxed_slo_goodput"])
    chosen = {}
    failed_static = st_train["_meta"]["failed_methods"] or {}
    for exp, tpot in (("qwen_peak_goodput", 50), ("qwen_relaxed_slo_goodput", 100)):
        r = {m: v for m, v in st_train["experiments"][exp]["results"].items()
             if m.startswith("static_budget_") and m not in failed_static}
        if not r:
            raise SystemExit(f"static-budget sweep produced no valid result: {failed_static}")
        best = max(r, key=lambda m: (r[m]["peak_goodput_gain_pct"], r[m]["mean_goodput_gain_pct"]))
        chosen[tpot] = int(best.rsplit("_", 1)[1])
    (RUNS / "static_selection.json").write_text(json.dumps({"selected_budgets_by_tpot_slo": chosen,
        "train_results": {e: st_train["experiments"][e]["results"] for e in st_train["experiments"]}}, indent=2))
    brute = {"bruteforce_static_budget": V + ["static_budget", "--budgets", f"50:{chosen[50]},100:{chosen[100]}"]}

    # 2. Train and held-out test scores for every audited attempt.
    audit = {**REFERENCE_SET, **VARIANT_SET, **SHORTCUT_SET, **brute}
    out_train = write_outputs(RUNS / "outputs_audit_train", {k: v for k, v in audit.items() if k in TRAIN_AUDIT})
    tr = run_eval("eval/train/evaluate.py", out_train, RUNS / "scores_train_audit.json")
    out_audit = write_outputs(RUNS / "outputs_audit", audit, RAW_MANIFESTS)
    te = run_eval("eval/test/evaluate.py", out_audit, RUNS / "scores_test_audit.json")

    # 3. Tamper check: weaken public simulator + baselines in a workspace copy.
    tamper_scores = RUNS / "scores_test_tamper.json"
    if not tamper_scores.exists():
        copy = RUNS / "tamper_workspace"
        if copy.exists():
            shutil.rmtree(copy)
        copy.mkdir()
        for d in ("data", "method", "baseline", "eval", "scoring"):
            shutil.copytree(ROOT / d, copy / d, ignore=shutil.ignore_patterns("downloads", "__pycache__"))
        sim = copy / "data" / "serving_sim.py"
        sim.write_text(sim.read_text().replace('"b": 0.15,', '"b": 0.015,').replace('"a": 8.0,', '"a": 0.8,'))
        base = copy / "baseline" / "schedulers.py"
        base.write_text(base.read_text().replace("SARATHI_BUDGET = {50.0: 256, 100.0: 512}", "SARATHI_BUDGET = {50.0: 16, 100.0: 16}"))
        out_t = write_outputs(copy / "outputs", {"fairbatching_pab": REFERENCE_SET["fairbatching_pab"]})
        run_eval(str(copy / "eval" / "test" / "evaluate.py"), out_t, tamper_scores, cwd=copy)
    tamper = json.loads(tamper_scores.read_text())

    # 4. Determinism: the audit's independent re-run of the reference must
    #    match the official scoring/scores_test.json exactly (checked below).

    # 5. Analysis.
    official = json.loads((ROOT / "scoring" / "scores_test.json").read_text())
    ref_vec = {e: official["experiments"][e]["results"]["fairbatching_pab"] for e in official["experiments"]}
    te_res = results(te)
    cohort = [m for m in te_res["qwen_peak_goodput"] if m not in RAW_MANIFESTS]
    ranks = rank_table(te, reference, ref_vec, sorted(te_res["qwen_peak_goodput"]))
    ranks_quality = rank_table(te, reference, ref_vec,
                               [m for m in cohort if not m.startswith("shortcut_")])
    tr_res = results(tr)
    tr_ref = {e: tr_res[e]["fairbatching_pab"] for e in tr_res}
    ranks_train = rank_table(tr, reference, tr_ref, sorted(m for m in tr_res["qwen_peak_goodput"] if m in TRAIN_AUDIT))
    noise = {name: bootstrap_paired(te, name, "fairbatching_pab")
             for name in ["better_margin_pab", "better_shed_pab", "better_srpt_shed_pab",
                          "better_exact_cost_pab", "fairbatching", "weaker_fixed_batch_pab"]}
    analysis = {
        "reference_vector_test": ref_vec,
        "test_results": te_res,
        "train_results": tr_res,
        "ranks_test_all_cohort": ranks,
        "ranks_test_quality_cohort": ranks_quality,
        "ranks_train_all_cohort": ranks_train,
        "failed_methods_test": te["_meta"]["failed_methods"],
        "failed_methods_train": tr["_meta"]["failed_methods"],
        "paired_bootstrap_vs_reference_test": noise,
        "tamper_check": {
            "fairbatching_pab_identical": results(tamper)["qwen_peak_goodput"]["fairbatching_pab"]
            == te_res["qwen_peak_goodput"]["fairbatching_pab"]
            and all(results(tamper)[e]["fairbatching_pab"] == te_res[e]["fairbatching_pab"] for e in te_res),
            "tampered_results": results(tamper),
        },
        "determinism_check": {
            "audit_rerun_identical_to_official": all(te_res[e]["fairbatching_pab"] == ref_vec[e] for e in ref_vec),
        },
        "static_budget_selection": chosen,
    }
    (AUD / "analysis.json").write_text(json.dumps(analysis, indent=2))
    log("ranks (test, all-audit cohort):")
    for m, r in sorted(ranks["ranks"].items(), key=lambda kv: -kv[1]):
        log(f"  {m:32s} {r:+.4f}")
    log("tamper identical: %s; determinism: %s" % (analysis["tamper_check"]["fairbatching_pab_identical"],
                                                    analysis["determinism_check"]))


if __name__ == "__main__":
    main()
