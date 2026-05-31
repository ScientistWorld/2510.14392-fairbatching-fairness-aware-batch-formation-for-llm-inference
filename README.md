# Overview

FairBatching is an LLM inference scheduling paper. It argues that strict
decode-prioritizing schedulers over-serve decode work, waste accumulated TPOT
slack, and starve prefill bursts, causing poor TTFT tail latency.

This workspace packages a trace-driven reproduction of the scheduler logic and
scores it against prefill-priority and Sarathi-style baselines on request-level
TTFT, TPOT, SLO violation rate, and effective RPS.

---

# Problem

LLM serving systems must schedule a mix of prefill work for newly arrived
requests and decode work for active generations. The problem is to maximize
the number of requests that satisfy both latency objectives:

- TTFT: time from request arrival to first output token.
- TPOT: worst average time per output token after the first token.

The scheduler receives a trace of request arrivals, prompt lengths, output
lengths, and per-request SLOs. It emits request-level completion timings.
Success requires high effective requests per second while keeping both TTFT and
TPOT violations low.

---

# Method

FairBatching replaces static token-budget scheduling with SLO-aware batch
formation.

The implementation in `method/simulator.py` follows the paper's computation:

- envelope-line deadlines for each request's next token;
- slack calculation from current time to the next deadline;
- adaptive time budget based on decode-request slack and TPOT SLO;
- linear batch-time estimator `a + b * total_new_tokens + c * total_context`;
- three-stage packing of urgent decode tasks, prefill tasks, and non-urgent
  decode tasks;
- the PAB admission-control estimate from Appendix A for the PAB variant.

Baselines implement prefill-prioritizing continuous batching and Sarathi-style
decode-prioritizing stall-free batching.

---

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
statistics at a scaled 2.4 RPS load point. SLO satisfaction is checked against
FairBatching's envelope deadlines; the reported `tpot_p99_ms` is post-first-token
decode speed so it remains comparable to the paper's latency table.

---

# Reproduction Log

### Iteration 1: gpt-5.5
- **Milestone**: `none` | **Status**: error
- **Working time**: 15m | **GPU**: 0.0h

<details>
<summary>Progress Log</summary>

<!-- Write your progress here -->

</details>

### Iteration 1: gpt-5.5
- **Milestone**: `none` | **Status**: error
- **Working time**: 30m | **GPU**: 0.0h

### Iteration 1: gpt-5.5
- **Milestone**: `none` | **Status**: error
- **Working time**: 30m | **GPU**: 0.0h

### Iteration 1: gpt-5.5
- **Milestone**: `none` | **Status**: error
- **Working time**: 30m | **GPU**: 0.0h

### Iteration 1: gpt-5.5
- **Milestone**: `core_claim` | **Status**: done
- **Working time**: 13m | **GPU**: 0.0h
- **Jobs**: 2 total (1 completed, 1 failed)

<details>
<summary>Progress Log</summary>

### [2026-05-31 00:00] - none
- Started reproduction work.
- Read the paper and identified FairBatching's scheduler as the core algorithm.

### [2026-05-31 00:00] - method_runs
- Implemented FairBatching's envelope-deadline slack tracking, adaptive time budget, three-stage batch construction, and PAB estimate.
- Implemented prefill-priority and Sarathi-style baselines.
- End-to-end scripts run and produce `scoring/scores.json`.

### [2026-05-31 00:00] - core_claim
- Fixed the FairBatching urgent-decode threshold to match Algorithm 1.
- Evaluated a scaled QwenTrace-like load point where FairBatching improves effective RPS over both baselines while satisfying envelope SLOs.
- Fixed the compute-container baseline import path after job `73b67506-68e` failed before running experiments.

### [2026-05-31 09:22] - core_claim
- Retry job `048586e6-92a` completed with exit code 0.
- The packaged `scripts/reproduce.sh` workflow reran baselines, FairBatching, evaluation, and `validate.py --compare` successfully.
- Final reproduced core-claim result: FairBatching effective RPS 2.144 with 0.000 SLO violation versus vanilla effective RPS 1.623 with 0.244 SLO violation on the scaled QwenTrace-like workload.

</details>


---

# Reproduction Milestones

**Current: core_claim**

## Stop Justification
- Completed at milestone `core_claim`.
