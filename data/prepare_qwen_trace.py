"""Extract the QwenTrace head segment used by the FairBatching paper.

The paper evaluates on the head segment of the Qwen-Bailian anonymous trace
(To-C chat trace, ``qwen_traceA_blksz_16.jsonl``).  Restricting the trace to
text requests and keeping the first 2500 of them reproduces the paper's
Table 2 statistics (prompt avg 896 / p90 1776, output avg 377 / p90 752 vs.
the paper's 892 / 1776 and 377 / 742).

Source: https://github.com/alibaba-edu/qwen-bailian-usagetraces-anon
License: Apache-2.0.  Only timestamp, input length and output length are kept.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

HEAD_REQUESTS = 2500


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data/downloads/qwen_bailian/qwen_traceA_blksz_16.jsonl"),
    )
    parser.add_argument("--output", type=Path, default=Path("data/qwen_trace_head.csv"))
    parser.add_argument("--requests", type=int, default=HEAD_REQUESTS)
    args = parser.parse_args()

    rows = []
    with args.source.open() as f:
        for line in f:
            record = json.loads(line)
            if record.get("type") != "text":
                continue
            rows.append(
                (
                    float(record["timestamp"]),
                    int(record["input_length"]),
                    int(record["output_length"]),
                )
            )
            if len(rows) >= args.requests:
                break
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp_s", "input_length", "output_length"])
        for ts, inp, out in rows:
            writer.writerow([f"{ts:.3f}", inp, out])
    print(f"wrote {len(rows)} requests to {args.output}")


if __name__ == "__main__":
    main()
