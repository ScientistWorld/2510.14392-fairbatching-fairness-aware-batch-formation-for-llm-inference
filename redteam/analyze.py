"""Summarize saved test results; no simulation or evaluator execution.

Ranks are relative to the initially shipped test result, using the user's
ranking rule. The eventual ranking cohort is unknown; both cohorts here are
explicitly recorded rather than presented as official production ranks.
"""
from __future__ import annotations

import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent
EXPERIMENT = "single_node_qwen_trace"
TARGET = "fairbatching_pab"


def read(path):
    return json.loads(path.read_text())


def result(scores, method=TARGET):
    return scores.get("experiments", {}).get(EXPERIMENT, {}).get("results", {}).get(method, {})


def ranks(rows, anchor, specs, weight):
    scales = {}
    for metric in specs:
        values = [row[metric] for row in rows.values() if metric in row]
        scales[metric] = statistics.pstdev(values) if len(values) > 1 else 0.0
    values = {}
    for name, row in rows.items():
        total = 0.0
        for metric, spec in specs.items():
            if metric not in row or metric not in anchor or not scales[metric]:
                continue
            direction = 1.0 if spec["higher_is_better"] else -1.0
            total += weight * abs(spec["coefficient"]) * direction * (row[metric] - anchor[metric]) / scales[metric]
        values[name] = total
    return {"members": list(rows), "population_std": scales, "rank_vs_initial_test_reference": values}


def main():
    initial = read(ROOT / "state/initial_scores_test.json")
    reference = result(initial)
    specification = read(ROOT / "state/initial_reference.json")["experiments"]
    spec = specification[EXPERIMENT]
    attempts, fresh = {}, {}
    for path in sorted(ROOT.glob("*/run.json")):
        status = read(path)
        name = path.parent.name
        scores = read(path.parent / "scores_test.json")
        row = result(scores)
        assert status["test"]["command"] == ["bash", "scripts/evaluate_test.sh"]
        assert status["test_is_fresh_process"] and status["test_invocation_has_no_arguments"]
        assert (path.parent / "run.sh").exists(), name
        attempts[name] = {
            "test_exit_code": status["test"]["exit_code"],
            "scores_are_fresh": status["scores_are_fresh"],
            "candidate_present_in_results": bool(row),
            "metrics": row if status["scores_are_fresh"] else None,
            "stale_file_metrics": None if status["scores_are_fresh"] else row,
            "methods_on_disk": status["output_method_names"],
            "methods_in_results": sorted(scores["experiments"][EXPERIMENT]["results"]),
        }
        if status["scores_are_fresh"]:
            fresh[name] = row
    baselines = {name: result(initial, name) for name in ("vllm_vanilla", "vllm_sarathi")}
    audit_rows = dict(fresh, **baselines)
    quality_names = ["reference_seed_0", "reference_seed_1", "reference_seed_2", "better_variant", "weaker_serial_fifo"]
    quality_rows = {name: fresh[name] for name in quality_names}
    quality_rows.update(baselines)
    seed_spread = {key: max(fresh[f"reference_seed_{i}"][key] for i in range(3)) - min(fresh[f"reference_seed_{i}"][key] for i in range(3)) for key in reference}
    assert not any(seed_spread.values())
    assert all(fresh[f"reference_seed_{i}"] == reference for i in range(3))
    assert not fresh["shortcut_omit_failed_method"]
    for name in ("shortcut_zero_timings", "shortcut_negative_timings", "shortcut_copy_easy_record", "shortcut_compress_arrivals", "shortcut_fixed_simulator"):
        row = fresh[name]
        improvements = [(row[key] - reference[key]) * (1 if metric["higher_is_better"] else -1) for key, metric in spec["metrics"].items()]
        assert all(value >= 0 for value in improvements) and any(value > 0 for value in improvements), name
    for name in ("shortcut_cached_answers", "shortcut_no_pab_tie", "shortcut_inactive_cap_150", "shortcut_inactive_cap_200"):
        assert fresh[name] == reference, name
    summary = {
        "anchor": "redteam/state/initial_scores_test.json:single_node_qwen_trace/results/fairbatching_pab",
        "rank_rule": "sum weight * abs(coefficient) * better_direction * (value - initial_reference) / cohort_population_std; missing values skipped; zero std contributes zero",
        "reference_metrics": reference,
        "seed_spread_max_minus_min": seed_spread,
        "quality_cohort": ranks(quality_rows, reference, spec["metrics"], spec["weight"]),
        "all_audit_cohort": ranks(audit_rows, reference, spec["metrics"], spec["weight"]),
        "positive_weight_experiments_missing_from_default_scores": {name: {"weight": s["weight"], "metrics": list(s["metrics"])} for name, s in specification.items() if s["weight"] > 0 and name not in initial["experiments"]},
        "attempts": attempts,
    }
    (ROOT / "analysis.json").write_text(json.dumps(summary, indent=2) + "\n")
    for cohort in ("quality_cohort", "all_audit_cohort"):
        print(cohort, "members:", len(summary[cohort]["members"]))
        for name, value in sorted(summary[cohort]["rank_vs_initial_test_reference"].items(), key=lambda item: -item[1]):
            print(f"  {name}: {value:+.9f}")
    print("Validated", len(attempts), "saved attempts; all hidden invocations have no arguments.")


if __name__ == "__main__":
    main()
