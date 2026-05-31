# Experiments

## single_node_qwen_trace

This experiment checks whether a single-node scheduler can serve a bursty
QwenTrace-like workload while satisfying both first-token and per-token latency
objectives. It compares request goodput and tail latency across prefill-first,
decode-first, and FairBatching-style schedulers.
