"""Rerun affected cases after eliminating unavailable reference coefficients.

The real scheduling search was already run. Its saved public train metrics
select the same 125-ms cap without accessing scoring/reference.json. Repeat
that candidate and the entire 20-configuration simple search, including the
original hidden test after each, to preserve direct evidence for this rule.
"""
import json
from pathlib import Path

from redteam.run_case import EXPERIMENT, PUBLIC_SELECTION_RULE, reset, run, select_public


def main():
    root = Path("redteam")
    names = [f"quality_cap_{cap}" for cap in (50, 75, 100, 125)]
    names += ["shortcut_inactive_cap_150", "shortcut_inactive_cap_200", "quality_raw_100", "quality_raw_200"]
    rows = {}
    for name in names:
        rows[name] = json.loads((root / name / "scores_train.json").read_text())["experiments"][EXPERIMENT]["results"]["fairbatching_pab"]
    best = select_public(rows, names)
    config = {"kind": "raw" if best.startswith("quality_raw_") else "capped", "cap": int(best.rsplit("_", 1)[1])}
    assert config == {"kind": "capped", "cap": 125}, config
    (root / "better_variant/config.json").write_text(json.dumps(config, indent=2))
    (root / "better_variant/selection.json").write_text(json.dumps({"selected_from_public_train_only": best, "selection_rule": PUBLIC_SELECTION_RULE, "train_metrics": rows}, indent=2))
    records = []
    try:
        for name in ("better_variant", "shortcut_static_sweep"):
            records.append(run(name))
    finally:
        reset()
        (root / "public_selection_summary.json").write_text(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
