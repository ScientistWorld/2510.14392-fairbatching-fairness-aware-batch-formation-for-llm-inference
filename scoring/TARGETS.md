# Targets

Primary targets:

- increase `effective_rps` on `single_node_qwen_trace`;
- increase `goodput_improvement_pct` on `qwen_slo_1000_100`, the relaxed-SLO
  QwenTrace-like setting with TTFT=1000ms and TPOT=100ms.

Constraint targets:

- keep `slo_violation_rate` low;
- keep `ttft_p99_ms` low enough that newly arrived requests are not starved;
- keep `envelope_tpot_p99_ms` at or below the TPOT SLO used by the benchmark;
- track raw `tpot_p99_ms` on the QwenTrace-like experiment as the paper-style
  post-first-token delivery constraint; the deadline-based TPOT metric is the
  benchmark's SLO-compliance view, while raw TPOT shows any remaining mismatch
  with the latency table.
- track `ttft_slo_violation_rate`, `envelope_tpot_slo_violation_rate`, and P95
  tail metrics so methods cannot hide broad SLO regressions behind one P99
  value.

Every target above is represented in `scoring/reference.json` and the current
`scoring/scores.json`. Additional synthetic traces are available in `data/` for
exploratory checks, but only experiments present in `scoring/scores.json` are
part of the default target set.
