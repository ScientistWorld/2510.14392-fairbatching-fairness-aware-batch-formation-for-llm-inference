#!/usr/bin/env bash
# Fetch the public workload trace used by this gym and rebuild the derived
# request table.  Idempotent; needs internet only when the trace is missing.
#
# Source: Qwen-Bailian anonymous usage traces (Apache-2.0),
#   https://github.com/alibaba-edu/qwen-bailian-usagetraces-anon
# Download size: ~56 MB (qwen_traceA_blksz_16.jsonl).
set -euo pipefail

cd "$(dirname "$0")/.."
DEST=data/downloads/qwen_bailian
FILE="$DEST/qwen_traceA_blksz_16.jsonl"
SHA256=07cedc9ed8aff301994ac68ed4aede8123b7603673575eeba9dd677de663db17
URL=https://media.githubusercontent.com/media/alibaba-edu/qwen-bailian-usagetraces-anon/main/qwen_traceA_blksz_16.jsonl

mkdir -p "$DEST"
touch data/downloads/.gitkeep
if [[ ! -s "$FILE" ]] || ! echo "$SHA256  $FILE" | sha256sum -c --status; then
    curl -fL --retry 3 -o "$FILE.part" "$URL"
    mv "$FILE.part" "$FILE"
fi
echo "$SHA256  $FILE" | sha256sum -c

# data/qwen_trace_head.csv (tracked) is the first 2500 text requests of the
# trace; regenerate it and check it matches the tracked copy.
python data/prepare_qwen_trace.py --source "$FILE" --output data/downloads/qwen_trace_head.csv
cmp data/downloads/qwen_trace_head.csv data/qwen_trace_head.csv && echo "data/qwen_trace_head.csv verified"
