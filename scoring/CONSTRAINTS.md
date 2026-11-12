# Constraints

- A method is a scheduler program registered under `outputs/schedulers/`
  that speaks the protocol in `briefing/evaluation.md`.  It decides admissions
  and batches only; the evaluator owns the workload, SLOs, clock, hardware
  cost and every reported number.
- Decisions must be physically valid: one decode token per running request
  per step, prefill chunks within the remaining prompt, at most 8192 new
  tokens per step, refusals only before a request's prefill starts.  Any
  invalid decision, crash or stall fails the run, which scores the worst value
  of every metric in every experiment.
- Refused requests count as SLO failures.  Latency tails are measured over
  admitted requests, but only when SLO attainment is at least that of the
  weaker baseline.
- The scheduler sees only requests that have already arrived and never a
  request's output length.  Do not read the evaluator's files (`eval/`), the
  raw trace, or the train seeds at run time to predict arrivals or lengths;
  do not write into `scoring/` or `eval/`.
- Do not change `data/qwen_trace_head.csv`, the SLOs, the load points or the
  hardware model: the held-out evaluator uses its own copies, so such changes
  only make the train numbers misleading.
- `eval/` must not import from `method/`.
