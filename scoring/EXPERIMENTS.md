# Experiments

## single_node_qwen_trace

This experiment checks whether a single-node scheduler can serve a bursty
QwenTrace-like workload while satisfying both first-token and per-token latency
objectives. It compares request goodput and tail latency across multiple
scheduler output artifacts at a scaled 2.4 RPS generator load point.

## qwen_slo_1000_100

This experiment checks whether the same single-node workload remains useful
when latency objectives are relaxed to TTFT=1000ms and TPOT=100ms. It scores
goodput improvement over the best baseline while still reporting SLO violation
rates and latency tails, so a method cannot improve only by ignoring the
latency constraints.
