# Evaluation

The reusable evaluator reads JSON artifacts produced by schedulers in
`outputs/`. It does not import or execute code from `method/`.

Metrics:

- `effective_rps`: offered trace request rate multiplied by the fraction of
  requests satisfying SLO, matching the paper's goodput definition.
- `slo_violation_rate`: fraction of requests violating TTFT or TPOT.
- `ttft_p99_ms`: 99th percentile time to first token.
- `tpot_p99_ms`: 99th percentile worst per-request TPOT.
- `envelope_tpot_p99_ms`: 99th percentile TPOT under the monotonic envelope
  deadline accounting used by the evaluator.

The default experiments are `single_node_qwen_trace`, a single-node workload
using a synthetic trace shaped from the paper's QwenTrace aggregate length and
SLO statistics at a scaled 2.4 RPS generator load point, and
`qwen_slo_1000_100`, a relaxed-SLO variant with TTFT=1000ms and TPOT=100ms.
SLO satisfaction is checked against request TTFT and deadline-envelope TPOT.
The reported `tpot_p99_ms` is post-first-token decode speed, so it remains
comparable to the paper's latency table while the envelope metric captures
deadline compliance.
