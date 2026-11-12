# Targets

Every target below is a scored metric in `scoring/reference.json` and is
written for every scheduler to `scoring/scores_train.json` /
`scoring/scores_test.json`.

## Benefit targets

- `qwen_peak_goodput.peak_goodput_gain_pct` (primary): raise peak goodput over
  the best baseline (paper, Qwen3-14B / QwenTrace / 500-50 ms: +112.6% for
  FairBatching-PAB; FairBatching without admission control +15.4%, which this
  simulator reproduces only in direction, kept as context).
- `qwen_peak_goodput.mean_goodput_gain_pct`: keep goodput high across the whole
  load sweep, including overload.
- `qwen_relaxed_slo_goodput.peak_goodput_gain_pct` (primary) and
  `.mean_goodput_gain_pct`: the same under TTFT 1000 / TPOT 100 ms (paper:
  +8.1% / +64.9%).
- `qwen_latency_detail.ttft_p99_ms` (primary): lower first-token tail latency
  at 2.5 RPS (paper Table 4: 1842 / 4585 / 1998 / 504 ms for vLLM-vanilla /
  vLLM-sarathi / FairBatching / FairBatching-PAB; the tuned Sarathi here has a
  smaller tail than the paper's, kept as context).

## Constraint targets

- `qwen_latency_detail.slo_attainment`: fraction of all offered requests
  (refusals included) that meet both SLOs; a gate sends every latency metric
  to its worst value if it drops below the weaker baseline.
- `qwen_latency_detail.tpot_p99_ms`: keep the per-token output rate within the
  50 ms objective (paper: 614 / 49 / 50 / 50 ms).
- `qwen_latency_detail.ttft_p50_ms`: do not trade typical first-token latency
  for the tail.
- Goodput itself is a constraint metric for refusal-based load control:
  refused requests count as failures in every goodput number.
