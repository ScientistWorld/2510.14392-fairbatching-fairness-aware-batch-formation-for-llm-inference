# Experiments

## single_node_qwen_trace

This experiment checks whether a single-node scheduler can serve a bursty
QwenTrace-like workload while satisfying both first-token and per-token latency
objectives. It compares request goodput and tail latency across multiple
scheduler output artifacts at a scaled 2.4 RPS generator load point.
