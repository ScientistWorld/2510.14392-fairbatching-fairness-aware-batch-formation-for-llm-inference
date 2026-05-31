# Progress

## What Works

Implemented a deterministic trace-driven simulator for FairBatching's scheduler,
including envelope deadlines, slack tracking, adaptive time budgets, three-stage
batch formation, and PAB admission estimation. Baselines for prefill-priority
and Sarathi-style decode-priority scheduling run through the same simulator.

The end-to-end scripts generate traces, run baselines and method variants, and
write `scoring/scores.json`. `python validate.py --compare` passes.

Compute job `048586e6-92a` verified the packaged workflow in the container with
exit code 0 after the baseline import-path fix.

## Results

Latest audited run on `single_node_qwen_trace`:

- vLLM-vanilla: effective RPS 1.623, SLO violation 0.244, TTFT P99 168.5 ms, TPOT P99 119.5 ms.
- vLLM-sarathi: effective RPS 1.573, SLO violation 0.261, TTFT P99 4640.7 ms, TPOT P99 50.0 ms.
- FairBatching: effective RPS 2.144, SLO violation 0.000, TTFT P99 174.9 ms, TPOT P99 160.1 ms.
- FairBatching-PAB: effective RPS 2.144, SLO violation 0.000, TTFT P99 174.9 ms, TPOT P99 160.1 ms.

This supports the core claim on a scaled trace: FairBatching improves effective
RPS while maintaining the envelope SLO that combines TTFT and TPOT deadlines.

## Remaining

Add additional load points and model configurations to move beyond the core
single-node claim.

## Issues

The synthetic workload is still a simulator rather than a vLLM deployment, so
absolute latency values should be interpreted as scaled reproduction numbers.

Job `73b67506-68e` failed because `baseline/run_baseline.py` could not import
`method.simulator` in the container's clean Python environment. Fixed
`scripts/baseline.sh` to set `PYTHONPATH` to the workspace root before running
baseline scripts. Retry job `048586e6-92a` completed successfully and reproduced
the scores above.

## Deviations from Paper

The paper evaluates production traces and a modified vLLM engine on large GPU
models. This reproduction uses a deterministic discrete-event simulator and
synthetic QwenTrace-like traces because production traces and the paper's vLLM
implementation are not released. The default trace uses a 2.4 RPS scaled load
point instead of the paper's exact sweep. The scheduler computation itself is
implemented from the paper, not replaced with a learned or hardcoded surrogate.
