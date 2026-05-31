# Targets

Primary target: increase `effective_rps` on `single_node_qwen_trace`.

Secondary target: keep the same scoring interface working on
`single_node_burstgpt_trace`.

Constraint targets:

- keep `slo_violation_rate` low;
- keep `ttft_p99_ms` low enough to avoid Sarathi-style prefill starvation;
- keep `envelope_tpot_p99_ms` at or below the TPOT SLO used by the scheduler;
- track raw `tpot_p99_ms` on the QwenTrace-like experiment as the paper-style
  post-first-token delivery constraint; the envelope metric is the scheduler's
  monotonic SLO formulation, while raw TPOT shows any remaining mismatch with
  the latency table.

Every target above is represented in `scoring/reference.json` and
`scoring/scores.json`.
