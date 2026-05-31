# Targets

Primary target: increase `effective_rps` on `single_node_qwen_trace`.

Constraint targets:

- keep `slo_violation_rate` low;
- keep `ttft_p99_ms` low enough to avoid Sarathi-style prefill starvation;
- keep `envelope_tpot_p99_ms` at or below the TPOT SLO used by the scheduler;
- track raw `tpot_p99_ms` on the QwenTrace-like experiment as the paper-style
  post-first-token delivery constraint; the envelope metric is the scheduler's
  monotonic SLO formulation, while raw TPOT shows any remaining mismatch with
  the latency table.
- track `ttft_slo_violation_rate`, `envelope_tpot_slo_violation_rate`, and P95
  tail metrics so methods cannot hide broad SLO regressions behind one P99
  value.

Every target above is represented in `scoring/reference.json` and the current
`scoring/scores.json`. The paper's BurstGPT numbers remain in
`scoring/reference.json` for context, but the synthetic BurstGPT sanity check is
not part of the default target set because its reproduced effective RPS is more
than 2x away from the paper's reported scale.
