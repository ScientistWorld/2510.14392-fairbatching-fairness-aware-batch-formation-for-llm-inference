# Targets

Primary target: increase `effective_rps` on `single_node_qwen_trace`.

Constraint targets:

- keep `slo_violation_rate` low;
- keep `ttft_p99_ms` low enough to avoid Sarathi-style prefill starvation;
- keep `tpot_p99_ms` low enough to avoid prefill-priority decode starvation.

Every target above is represented in `scoring/reference.json` and
`scoring/scores.json`.
