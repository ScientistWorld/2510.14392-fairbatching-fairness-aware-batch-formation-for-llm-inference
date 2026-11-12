"""Small follow-up: successful omission of a measured failed method and ties."""
import json
from pathlib import Path

from redteam.run_case import reset, run


def main():
    records = []
    try:
        for name in ("shortcut_omit_failed_method", "shortcut_inactive_cap_150", "shortcut_inactive_cap_200"):
            records.append(run(name))
    finally:
        reset()
        Path("redteam/followup_summary.json").write_text(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
