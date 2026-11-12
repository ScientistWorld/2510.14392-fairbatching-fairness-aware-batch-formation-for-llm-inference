"""One CPU job for all audit experiments; restore all non-audit artifacts."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from redteam.run_case import EXPERIMENT, PUBLIC_SELECTION_RULE, reset, run, select_public


def main():
    records = []
    try:
        for seed in (0, 1, 2):
            records.append(run(f"reference_seed_{seed}"))
        records.append(run("weaker_serial_fifo"))
        quality = [f"quality_cap_{cap}" for cap in (50, 75, 100, 125)]
        quality += ["quality_raw_100", "quality_raw_200"]
        for name in quality:
            records.append(run(name))
        for cap in (150, 200):
            records.append(run(f"shortcut_inactive_cap_{cap}"))
        ref = json.loads(Path("redteam/reference_seed_0/scores_train.json").read_text())["experiments"][EXPERIMENT]["results"]["fairbatching_pab"]
        rows = {"reference": ref}
        for name in quality:
            rows[name] = json.loads((Path("redteam") / name / "scores_train.json").read_text())["experiments"][EXPERIMENT]["results"]["fairbatching_pab"]
        best = select_public(rows, quality)
        config = {"kind": "raw" if best.startswith("quality_raw") else "capped", "cap": int(best.rsplit("_", 1)[1])}
        Path("redteam/better_variant/config.json").write_text(json.dumps(config, indent=2))
        Path("redteam/better_variant/selection.json").write_text(json.dumps({"selected_from_public_train_only": best, "selection_rule": PUBLIC_SELECTION_RULE, "train_metrics": rows}, indent=2))
        records.append(run("better_variant"))
        names = [
            "shortcut_static_sweep", "shortcut_zero_timings", "shortcut_negative_timings",
            "shortcut_copy_easy_record", "shortcut_relax_slos", "shortcut_compress_arrivals",
            "shortcut_fixed_simulator", "shortcut_cached_answers", "shortcut_no_pab_tie",
            "shortcut_summary_flags", "shortcut_direct_scores", "shortcut_empty_records",
            "shortcut_null_timings", "shortcut_omit_unregistered_method", "shortcut_fail_task_fast_tails",
            "shortcut_omit_failed_method",
        ]
        for name in names:
            records.append(run(name))
    finally:
        reset()
        Path("redteam/job_summary.json").write_text(json.dumps(records, indent=2))
        print("Audit job finished; non-audit experiment files restored.", flush=True)


if __name__ == "__main__":
    main()
