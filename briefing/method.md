# Method

FairBatching (Lyu et al., 2025) replaces static token-budget, decode-first
scheduling with SLO-aware batch formation.  `method/fairbatching.py` is a
scheduler program implementing the paper's computational steps:

- **Envelope deadlines (Section 3.1):** token `j` of request `i` is due at
  `arrival_i + ttft_slo + tpot_slo * j`; a request's slack is its next token's
  deadline minus the current time.
- **Adaptive time budget (Section 3.2):** the next batch may take
  `max(min decode slack, min decode TPOT SLO)`; step time is estimated with the
  linear model `a + b * new_tokens + c * context` (the evaluator's published
  profile).  With no decode running, only the token budget applies.
- **Fair batch formation (Section 3.3, Algorithm 1):** urgent decodes (slack
  below `budget + min TPOT SLO`), then prefills, then the remaining decodes,
  each group by ascending slack; items are packed while they fit the remaining
  time and token budget, and a prefill that does not fit is chunked.
- **Prefill admission budget (Section 3.4, Appendix A; `--variant pab`):** an
  arriving request is rejected when the PAB computed over all admitted
  requests is smaller than its prompt length.

`fairbatching` is FairBatching-vanilla, `fairbatching_pab` FairBatching-PAB
(the reference every attempt is ranked against).

Baselines (`baseline/schedulers.py`): vLLM-vanilla (prefill-first continuous
batching with an 8192-token budget) and vLLM-sarathi (stall-free batching:
all decodes, then FCFS prefill chunks under a token budget tuned on the train
split per SLO setting: 256 tokens at TPOT 50 ms, 512 at TPOT 100 ms).
