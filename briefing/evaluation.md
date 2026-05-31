# Evaluation

The reusable evaluator reads JSON artifacts produced by schedulers in
`outputs/`. It does not import or execute code from `method/`.

Metrics:

- `effective_rps`: SLO-satisfied requests divided by trace makespan.
- `slo_violation_rate`: fraction of requests violating TTFT or TPOT.
- `ttft_p99_ms`: 99th percentile time to first token.
- `tpot_p99_ms`: 99th percentile worst per-request TPOT.

The main experiment is `single_node_qwen_trace`, a single-node workload using a
synthetic trace shaped from the paper's QwenTrace aggregate length and SLO
statistics.
