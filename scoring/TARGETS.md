# Targets

Primary target: increase `effective_rps` on `single_node_qwen_trace`.

Constraint targets:

- keep `slo_violation_rate` low;
- keep `ttft_p99_ms` low enough to avoid Sarathi-style prefill starvation;
- keep `envelope_tpot_p99_ms` at or below the TPOT SLO used by the scheduler;
- track raw `tpot_p99_ms` on the QwenTrace-like experiment as a diagnostic for
  post-first-token delivery smoothness.

Every target above is represented in `scoring/reference.json` and
`scoring/scores.json`.
