# Experiments

## single_node_qwen_trace

This experiment checks whether a single-node scheduler can serve a bursty
QwenTrace-like workload while satisfying both first-token and per-token latency
objectives. It compares request goodput and tail latency across prefill-first,
decode-first, and FairBatching-style schedulers at a scaled 2.4 RPS load point.

## single_node_burstgpt_trace

This experiment repeats the same single-node scoring interface on a
BurstGPT-like request length distribution. It is a lighter sanity check for
cross-trace portability rather than the main core-claim benchmark.
