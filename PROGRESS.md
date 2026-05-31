# Progress

## What Works

Implemented a deterministic trace-driven simulator for FairBatching's scheduler,
including envelope deadlines, slack tracking, adaptive time budgets, three-stage
batch formation, and PAB admission estimation. Baselines for prefill-priority
and Sarathi-style decode-priority scheduling run through the same simulator.

The end-to-end scripts generate traces, run baselines and method variants, and
write `scoring/scores.json`. `python validate.py --compare` passes.

## Results

Latest local smoke test on `single_node_qwen_trace`:

- vLLM-vanilla: effective RPS 1.088, SLO violation 0.500, TTFT P99 177.7 ms, TPOT P99 104.9 ms.
- vLLM-sarathi: effective RPS 1.418, SLO violation 0.339, TTFT P99 6165.4 ms, TPOT P99 50.0 ms.
- FairBatching: effective RPS 1.018, SLO violation 0.528, TTFT P99 5853.3 ms, TPOT P99 87.3 ms.
- FairBatching-PAB: effective RPS 1.195, SLO violation 0.500, TTFT P99 1498.3 ms, TPOT P99 100.0 ms.

These results show the scheduler executes and exposes the intended trade-offs,
but they do not yet reproduce the paper's effective-RPS improvement.

## Remaining

Tune the simulator workload and linear step-time parameters so that the
simplified environment better matches the paper's vLLM regime. Add additional
load points after the core single-node claim is supported.

## Issues

The current synthetic workload is sensitive to overload. At moderate load,
FairBatching behaves similarly to prefill-priority scheduling; at high load,
the simplified PAB estimator is not yet strong enough to recover the paper's
goodput gains.

## Deviations from Paper

The paper evaluates production traces and a modified vLLM engine on large GPU
models. This reproduction uses a deterministic discrete-event simulator and
synthetic QwenTrace-like traces because production traces and the paper's vLLM
implementation are not released. The scheduler computation itself is implemented
from the paper, not replaced with a learned or hardcoded surrogate.
