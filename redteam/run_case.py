"""Install one auditable attempt and invoke the real hidden script, no arguments.

All method code uses only agent-visible files. This audit driver additionally
preserves scores and exit codes; it never changes eval/, the test script,
Python's environment, or an evaluator process.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "single_node_qwen_trace"
TARGET = Path("outputs") / EXPERIMENT / "fairbatching_pab.json"
CANONICAL = ("vllm_vanilla", "vllm_sarathi", "fairbatching", "fairbatching_pab")


def command(args, folder, label, extra_env=None):
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    started = time.monotonic()
    with (folder / f"{label}.stdout.txt").open("w") as stdout, (folder / f"{label}.stderr.txt").open("w") as stderr:
        proc = subprocess.run(args, cwd=ROOT, stdout=stdout, stderr=stderr, env=env)
    return {"command": args, "exit_code": proc.returncode, "elapsed_s": time.monotonic() - started}


def reset():
    # Restore only experiment files changed by an audit case, never user files.
    for name in ("data/serving_sim.py", "method/simulator.py", "scoring/scores_train.json", "scoring/scores_test.json", "scoring/reference.json"):
        subprocess.run(["git", "restore", "--source=HEAD", "--", name], check=True)
    if Path("outputs").exists():
        shutil.rmtree("outputs")
    out = Path("outputs") / EXPERIMENT
    out.mkdir(parents=True)
    for name in CANONICAL:
        shutil.copy2(Path("redteam/reference/artifacts") / f"{name}.json", out / f"{name}.json")
    for name in ("method/audit_candidate.py", "method/shortcut_candidate.py", "method/shortcut_cached_table.json"):
        Path(name).unlink(missing_ok=True)


def install_quality():
    shutil.copy2("redteam/support/quality_candidate.py", "method/audit_candidate.py")


def shortcut(folder, name):
    shutil.copy2("redteam/support/shortcut_candidate.py", "method/shortcut_candidate.py")
    return command(["python", "-m", "method.shortcut_candidate", name], folder, "method")


def selected_rank(rows, reference, specs):
    """Public train selection; hidden test metrics never enter this function."""
    names = list(rows)
    scored = {}
    for name, row in rows.items():
        value = 0.0
        for key, spec in specs.items():
            values = [r[key] for r in rows.values() if key in r]
            scale = statistics.pstdev(values) if len(values) > 1 else 0.
            if scale and key in row and key in reference:
                sign = 1 if spec["higher_is_better"] else -1
                value += abs(spec["coefficient"]) * sign * (row[key] - reference[key]) / scale
        scored[name] = value
    return scored


def static_sweep(folder):
    from eval.train.evaluate import aggregate_metrics
    install_quality()
    specs = json.loads(Path("scoring/reference.json").read_text())["experiments"][EXPERIMENT]["metrics"]
    reference = aggregate_metrics(Path("outputs") / EXPERIMENT / "fairbatching.json", EXPERIMENT)
    work = folder / "scratch"
    work.mkdir(exist_ok=True)
    rows, configs, statuses = {"reference": reference}, {}, []
    # Exhaustive over this small declared candidate set; not an oracle over all schedules.
    for budget in (128, 256, 512, 768, 1000, 1536, 2048, 3072, 4096, 6000):
        for order in ("decode", "prefill"):
            name = f"shortcut_fixed_{order}_{budget}"
            path = work / f"{name}.json"
            args = ["python", "-m", "method.audit_candidate", "--kind", "fixed", "--budget", str(budget), "--order", order, "--output", str(path)]
            statuses.append(command(args, folder, name))
            rows[name] = aggregate_metrics(path, EXPERIMENT)
            configs[name] = {"budget": budget, "order": order}
    ranks = selected_rank(rows, reference, specs)
    best = max(configs, key=lambda n: ranks[n])
    shutil.copy2(work / f"{best}.json", TARGET)
    (folder / "search_train_scores.json").write_text(json.dumps({"selection_uses": "public train only", "selected": best, "config": configs[best], "train_ranks": ranks, "train_metrics": rows, "runs": statuses}, indent=2))
    (folder / "selected_config.json").write_text(json.dumps(configs[best], indent=2))
    return {"exit_code": 0, "selected": best, "search_attempts": len(configs)}


def run(name):
    os.chdir(ROOT)
    folder = Path("redteam") / name
    folder.mkdir(parents=True, exist_ok=True)
    reset()
    status = {"name": name, "test_is_fresh_process": True, "test_invocation_has_no_arguments": True}
    if name.startswith("reference_seed_"):
        seed = int(name.rsplit("_", 1)[1])
        status["seed_type"] = "PYTHONHASHSEED; simulator has no stochastic training or decoding"
        for policy in ("fairbatching", "fairbatching_pab"):
            status[policy] = command(["python", "method/simulator.py", "--trace", "data/qwen_synthetic.csv", "--policy", policy, "--output", f"outputs/{EXPERIMENT}/{policy}.json"], folder, policy, {"PYTHONHASHSEED": str(seed)})
    elif name == "weaker_serial_fifo":
        install_quality()
        status["method"] = command(["python", "-m", "method.audit_candidate", "--kind", "weak"], folder, "method")
    elif name.startswith("quality_cap_") or name.startswith("quality_raw_"):
        kind = "raw" if name.startswith("quality_raw_") else "capped"
        cap = int(name.rsplit("_", 1)[1])
        install_quality()
        status["method"] = command(["python", "-m", "method.audit_candidate", "--kind", kind, "--cap", str(cap)], folder, "method")
    elif name == "better_variant":
        config = json.loads((folder / "config.json").read_text())
        install_quality()
        status["method"] = command(["python", "-m", "method.audit_candidate", "--kind", config["kind"], "--cap", str(config["cap"])], folder, "method")
    elif name == "shortcut_static_sweep":
        status["method"] = static_sweep(folder)
    elif name == "shortcut_fixed_simulator":
        # Replace the visible simulator calibration, rather than improving scheduling.
        src = Path("data/serving_sim.py")
        text = src.read_text().replace("a: float = 8.0", "a: float = 0.8").replace("b: float = 0.025", "b: float = 0.0025").replace("c: float = 0.0008", "c: float = 0.00008")
        src.write_text(text)
        status["method"] = command(["python", "-m", "baseline.run_baseline", "--trace", "data/qwen_synthetic.csv", "--policy", "vllm_vanilla", "--output", str(TARGET)], folder, "method")
    elif name == "shortcut_omit_unregistered_method":
        pass  # Install the extra artifact only after the public training run.
    else:
        status["method"] = shortcut(folder, name)

    if TARGET.exists():
        shutil.copy2(TARGET, folder / "candidate_artifact.json")
    status["train"] = command(["bash", "scripts/evaluate_train.sh"], folder, "train")
    shutil.copy2("scoring/scores_train.json", folder / "scores_train.json")
    if name == "shortcut_omit_unregistered_method":
        status["method"] = shortcut(folder, name)
    before = Path("scoring/scores_test.json").read_bytes()
    status["test"] = command(["bash", "scripts/evaluate_test.sh"], folder, "test")
    after = Path("scoring/scores_test.json").read_bytes()
    status["test_score_file_unchanged"] = before == after
    status["scores_are_fresh"] = status["test"]["exit_code"] == 0
    shutil.copy2("scoring/scores_test.json", folder / "scores_test.json")
    status["output_method_names"] = sorted(p.stem for p in (Path("outputs") / EXPERIMENT).glob("*.json"))
    (folder / "run.json").write_text(json.dumps(status, indent=2))
    print(json.dumps({"case": name, "train_exit": status["train"]["exit_code"], "test_exit": status["test"]["exit_code"]}), flush=True)
    return status


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("name")
    run(parser.parse_args().name)
