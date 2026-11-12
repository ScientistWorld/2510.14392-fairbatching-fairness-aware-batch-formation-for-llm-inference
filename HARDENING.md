# Hardening report (iteration 11)

**Verdict: better results now score higher and worse ones lower.** The gym was rebuilt around an
evaluator-owned simulation:

- **What a method submits:** a scheduler program, not timing records.
- **What the evaluator owns:** the workload (the paper's own QwenTrace), SLOs, clock, hardware cost
  model and every number.
- **The hidden test:** runs every registered scheduler on workloads that only `eval/test/` can
  generate.
- **Results:**
  - Four honest refinements of the reference beat it by far more than the paired noise.
  - Every weaker scheduler and every baseline scores below the reference.
  - Every shortcut the red team ran is now either impossible or scores the worst value.
  - One honest ablation ties the reference (see "Left as is").

All numbers below are from the final verification job (`c773562a-a1a`, exit 0, `validate.py`
passes). Evidence:
- `redteam/hardening/analysis.json`: ranks, per-metric results, paired bootstraps, tamper and
  determinism checks.
- `redteam/hardening/audit_runs/`: every evaluator output and log.
- `redteam/hardening/run_audit.py`: reruns everything and resumes after preemption.
- The attempts themselves: `redteam/hardening/variants.py` and `shortcuts.py`.

## 1. What changed and why

| Problem in REDTEAM.md | Fix |
|---|---|
| Evaluator aggregated method-written per-request timings. Zero, negative, copied or relaxed timings and compressed arrivals all paid. | A method is a program that speaks a JSON-lines step protocol. The evaluator simulates the engine, validates every decision and computes all latencies (`data/serving_sim.py`, test copy `eval/test/harness.py`). A crash, an invalid decision (unknown or duplicate request, oversize batch, more than one decode token, rejecting a started request), malformed output or a stall fails the run. |
| Test answers were a public trace, so a lookup table of reference outputs tied the reference. | Instances are sampled with undisclosed seeds inside `eval/test/`. A scheduler only sees requests that have already arrived and never their output lengths. |
| Simulator calibration lived in public code (`fixed_simulator` +0.174). | `eval/test/` is self-contained: its own simulator, workload copy (hash-checked, loaded before any scheduler starts) and frozen baselines. It imports nothing from `data/`, `method/` or `baseline/`. |
| Method whitelist came from the mutable train scores. Omitted or failed methods cost nothing, and failed runs wrote no scores. | Every manifest in `outputs/schedulers/` is scored, and the four shipped names are always emitted. Any failure gives the worst value of every metric in every experiment (gain −100%, attainment 0, TTFT 10 s, TPOT 1 s). |
| A positive-weight experiment (30%) was never emitted. | The experiment list is fixed in the evaluator; all three experiments are always written. |
| Primary metric was saturated: the reference served 100% at one light load point. PAB had no effect. | Hardware calibrated to the paper's Qwen3-14B/H20 regime. Peak goodput is measured over load sweeps up to overload (Figure 5 / Table 3), as gain over evaluator-owned baselines. |
| Raw TPOT went the wrong way: FB 167 ms vs vanilla 90 ms. | Scored TPOT is the per-token rate measured from the later of the first token and the TTFT deadline. This is the only definition consistent with every row of Table 4 (FB 50/50, Sarathi 46/49, vanilla 256/614 at p95/p99). Raw TPOT puts FB near 300 ms. SLO success is the paper's envelope rule (§3.1), unchanged. |
| Synthetic trace with a hand-written 36-request burst cycle and independent lengths. | Real data: the head of the Qwen-Bailian trace A, first 2500 text requests. It reproduces Table 2 (prompt 896/1776 vs 892/1776; output 377/752 vs 377/742). Real arrival windows are rescaled to each load. Length pairs are bootstrapped around the window, with random request ids. |
| Same ideal linear model for estimating and executing step times. | True step time adds prefill-attention cost and ±1% jitter. Schedulers only get the offline linear fit, whose 0.8% median error is comparable to the paper's ±1.3%, plus each step's measured duration. |
| Scheduler compute was unbounded, and a slow or stalled program could stall the hidden test. | Limits on the scheduler process's own CPU time: 300 s per run, 3600 s per evaluation; the reference uses ~1 s per run. Each message must be answered within 30 s. A per-run wall backstop of `max(600 s, 100 × CPU used)` catches sleepers. Once a method fails, its remaining runs are skipped. CPU-based limits cannot be tripped by a busy machine; an oversubscribed first job exposed this. |
| A running scheduler could overwrite evaluator data mid-evaluation. | The workload pool is read and hash-verified once, before any scheduler process starts. Score files are written last. |

**Experiments.** Weights follow what the paper says matters.

| Experiment | Weight | What it measures | Scored metrics (coefficient) |
|---|---|---|---|
| `qwen_peak_goodput` | 0.5 | Peak and mean effective-RPS gain over the best frozen baseline; TTFT 500 / TPOT 50 ms; loads 1–5 RPS; 12 trace windows replayed at every load. The single-node capacity headline. | peak gain (1.0), mean gain (0.5) |
| `qwen_latency_detail` | 0.25 | The Table 4 operating point (2.5 RPS, 12 windows). The TTFT-tail headline. | TTFT p99 (1.0), SLO attainment (1.0), TPOT p99 (0.5), TTFT p50 (0.25) |
| `qwen_relaxed_slo_goodput` | 0.25 | The Table 5 row TTFT 1000 / TPOT 100 ms; loads 1.5–6 RPS. SLO versatility. | peak gain (1.0), mean gain (0.5) |

In `qwen_latency_detail`, latency counts over admitted requests, as in Table 4. A gate gives every
metric of that experiment its worst value if SLO attainment falls below the weakest baseline.

Reference results in `scoring/reference.json` keep every paper number unchanged:
- `qwen_peak_goodput` uses Table 5's number for exactly this configuration (Qwen3-14B, QwenTrace,
  500/50): +112.6% for FairBatching-PAB.
- Under `_` keys, not as targets:
  - Table 3's four-model geometric means.
  - The two paper values this setup does not reach in magnitude (see §6).

## 2. Do better results score higher?

Rank = Σ weight × |coef| × direction × (attempt − reference) / std over the cohort. Two cohorts
are shown:
- **All:** 22 audited attempts.
- **Non-shortcut:** 14 attempts, which is closer to a population of honest attempts.

Before = REDTEAM.md on the old gym.

| Attempt (held-out test) | Peak gain % | Mean gain % | Attainment | TTFT p99 ms | TPOT p99 ms | Relaxed peak % | Rank (all) | Rank (non-shortcut) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| better: shed queued requests that can no longer meet TTFT + SRPT prefill order | 119.8 | 111.9 | 0.940 | 512 | 49.5 | 77.9 | **+0.291** | **+0.607** |
| better: shed + 6% estimation margin | 120.7 | 112.2 | 0.942 | 500 | 49.5 | 73.0 | **+0.280** | **+0.587** |
| better: 6% estimation margin | 114.4 | 109.5 | 0.937 | 553 | 49.5 | 69.1 | **+0.214** | **+0.443** |
| better: exact public hardware formula + 1.2% margin | 114.7 | 109.3 | 0.937 | 543 | 49.9 | 67.3 | **+0.208** | **+0.431** |
| **reference FairBatching-PAB** | 99.3 | 99.6 | 0.915 | 552 | 50.4 | 48.9 | 0 | 0 |
| weaker: FB-FB ablation (fixed 50 ms budget) + PAB | 102.5 | 103.2 | 0.919 | 557 | 50.0 | 46.3 | +0.022 | +0.007 |
| weaker: FairBatching-vanilla (no PAB) | 11.1 | 3.2 | 0.801 | 1639 | 51.3 | 5.4 | −1.148 | −2.610 |
| weaker: FB-FB ablation without PAB | 6.3 | −3.0 | 0.787 | 1719 | 50.9 | 3.4 | −1.217 | −2.849 |
| vLLM-sarathi (tuned budget) | 0.0 | 0.0 | 0.771 | 2071 | 49.6 | 0.0 | −1.288 | −3.167 |
| vLLM-vanilla | −0.1 | −1.6 | 0.753 | 1623 | 729.0 | −6.2 | −1.507 | −3.462 |

**Before vs. after:**
- The red team's better variant (125 ms batch cap) ranked **+0.063**, and only through latency
  tails; the primary metric was stuck at its bound.
- The weaker serial FIFO ranked −8.73.
- Now four honest improvements rank +0.21 to +0.29 (all cohort) / +0.43 to +0.61 (non-shortcut).
- FairBatching without PAB and both baselines rank clearly below the reference.
- The 6-window audit gave the same ordering (`analysis_6windows.json`).

**Paired noise.** Every method sees the same instances; this is a bootstrap over trace windows,
95% interval. Improvement over the reference:

| Variant | Peak gain (pts) | Mean gain (pts) | Attainment | Relaxed peak gain (pts) |
|---|---:|---:|---:|---:|
| shed + margin | +21.4 [18.3, 24.3] | +12.6 [11.7, 13.5] | +0.027 [0.021, 0.033] | +24.2 [17.9, 26.2] |
| margin only | +15.2 [12.9, 17.2] | +9.8 [9.2, 10.6] | +0.022 [0.016, 0.028] | +20.2 [14.2, 22.2] |

FairBatching-vanilla vs. the reference: −88.2 [−98.1, −76.7] peak points. All intervals exclude 0.

**Seed and determinism.**
- Simulations are deterministic: the audit's independent rerun of the reference reproduced
  `scoring/scores_test.json` bit-for-bit.
- Window sampling is the remaining noise. The reference's absolute peak goodput is 3.89 RPS on
  train vs 3.92 RPS on test.
- The gain metric is 89.9% on train vs 99.3% on test, because the baselines' peak (the
  denominator) is 2.05 vs 1.97 RPS. Moving from 6 to 12 windows shrank this gap, which was
  81.5% vs 110%.

## 3. Shortcuts (held-out test)

| Shortcut (REDTEAM.md) | Rank before | What it is now | Rank after |
|---|---:|---|---:|
| zero timings / negative timings / copy fastest record | +0.176 / +1.531 / +0.159 | Impossible: methods report no timings. The closest attack, printing a forged record file instead of decisions (`shortcut_forge_records`), fails the run. | −3.945 (worst) |
| relax SLOs in the artifact | +0.064 | Impossible: SLOs come from the evaluator's workload. | n/a |
| compress arrival span | +3.772 | Impossible: arrivals and offered rate come from the evaluator's workload. | n/a |
| cheaper simulator (`fixed_simulator`) | +0.174 | Copy of the workspace with `data/serving_sim.py` made 10× cheaper and `baseline/` Sarathi budget sabotaged to 16 tokens. Held-out reference scores are **bit-identical** (`tamper_check`). | 0 (no effect) |
| cached reference answers on the public trace | 0 (tie) | Held-out instances are not visible. The strongest form, an oracle that recognises each public train instance and picks, per instance, the configuration that scored best in an exact offline replay (`shortcut_train_oracle`), ranks **+1.06 on train** (peak gain 112.7 vs 89.9), but on the hidden test it can recognise nothing. | 0 (= reference, which it falls back to) |
| omit an unregistered / failed method | 0 | Every manifest on disk is scored and the shipped names are always emitted. Malformed manifest and crashing scheduler both score worst. | −3.945 |
| empty / null records (evaluator crashed, no fresh scores) | no score | Any invalid output fails only that method; the evaluator still writes a complete file. | −3.945 |
| write forged `scoring/scores_test.json` | −0.704 | The evaluator writes its file last. The attempt (which then runs the reference scheduler) scores exactly the reference. | 0 (no gain) |
| summary flags | −0.704 | Nothing a scheduler prints besides decisions is read. | n/a |
| fail the task with fast tails | −5.787 | Reject every request (`shortcut_reject_all`). | −3.945 |
| static-budget brute force | −0.753 | Budgets 128–8192 swept on train, best chosen per SLO (256 / 512 = tuned Sarathi). | −1.288 |
| PAB-less ablation tied PAB | 0 | Now clearly separated (FairBatching-vanilla −1.148). | −1.148 |
| new: admit only short prompts so admitted latency looks good | — | Attainment below the weakest baseline trips the latency gate; goodput −26%. | −3.044 |
| new: oversized batch, 4 decode tokens per step | — | Physically invalid, so the run fails. | −3.945 |
| new: stop reading input / answer slowly (deny a score) | — | Fails within 30 s / ≤600 s, then remaining runs are skipped. | −3.945 |

The two "inactive batch cap" ties were honest changes that did not touch the old trace. They are not
shortcuts, so I did not replay them.

## 4. Reference before and after

**Before (old gym, test split):** effective RPS 2.937 (at its bound), 0% violations, TTFT p99
167 ms, raw TPOT p99 167 ms. PAB = FB.

**After (`scoring/scores_test.json`):**

| Metric | Test | Train |
|---|---:|---:|
| Peak-goodput gain | +99.3% | +89.9% |
| Mean-goodput gain | +99.6% | — |
| Absolute peak | 3.92 RPS | 3.89 RPS |
| SLO attainment at 2.5 RPS | 0.915 | — |
| TTFT p50 / p99 | 141 / 552 ms | — |
| TPOT p99 | 50.4 ms | — |
| Relaxed-SLO peak gain | +48.9% | +51.9% |

**Paper direction holds everywhere** (paper → test):
- Peak-goodput gain at 500/50 ms: FB-PAB +112.6% → +99.3%; FB-vanilla +15.4% → +11.1%.
- Relaxed SLO at 1000/100 ms: FB-vanilla +8.1% → +5.4%; FB-PAB +64.9% → +48.9%.
- Table 4 pattern:
  - TTFT p99: vanilla 1623 / Sarathi 2071 / FB 1639 / FB-PAB 552 ms (paper 1842 / 4585 / 1998 / 504).
  - TPOT p99: 729 / 49.6 / 51.3 / 50.4 ms (paper 614 / 49 / 50 / 50).

## 5. Headroom check

The reference sits at the bound of **no** scored metric:

| Metric | Reference | Bound | Best variant |
|---|---:|---:|---:|
| Peak gain (main) | +99.3% | +154% (5 RPS offered / best-baseline peak 1.97) | +120.7% |
| Peak gain (relaxed) | +48.9% | +101.6% | +77.9% |
| SLO attainment | 0.915 | 1 | 0.942 |
| TTFT p99 | 552 ms | — | 500 ms |

What was added from where the paper's method falls short:
- Load beyond the reference's comfort zone (sweeps to 5 and 6 RPS).
- Step-time estimation error: a nonlinear true cost model with jitter.
- The relaxed-SLO setting, where the paper itself reports smaller gains.

All three are where the better variants gain. No single simple variant reaches a bound. The brute
force the paper replaces (static token budgets) scores far below the reference. The simple
baselines sit clearly below it, at −1.3 to −1.5.

## 6. Left as is, and what I could not fix

- **A scheduler that reads `eval/test/` at run time** could recover the hidden seeds and replay
  instances like the train oracle does. The scheduler runs as the same user, in the same
  container, as the evaluator, so this cannot be blocked from inside the gym. It needs the separate
  grader to flag reads of `eval/`.
- **The trace is public.** A scheduler could try to match arrival-gap patterns to the public Qwen
  trace to anticipate future arrivals. Lengths are resampled and ids randomised, so only arrival
  times could leak. Not attempted, and left.
- **Using the simulator's published hardware formula** (`better_exact_cost_pab`) gains no more than
  a plain 6% margin (+0.208 vs +0.214), so it is not a major shortcut. A better profiler is a
  legitimate improvement.
- **The FB-FB + PAB ablation ties the reference** (+0.022 / +0.007, and inside noise on attainment
  and the relaxed experiment). With PAB on, the paper's adaptive time budget adds little in this
  simulator. Without PAB the ablation is below FairBatching-vanilla, as in the paper's breakdown.
  This is an honest alternative the test barely separates, i.e. a headroom limit, not a loophole.
- **Magnitudes differ from the paper:**
  - FB-vanilla's peak gain is +11% (test) / +6% (train) vs +15.4%.
  - Tuned Sarathi's TTFT p99 is ~2.0 s vs 4.6 s.

  The direction and ordering hold. Both paper values are not on this setup's scale. Per the
  hardening rules, they are kept unchanged in `scoring/reference.json` under `_..._not_on_scale`
  keys rather than as targets, and `fairbatching` is listed as a comparison point: the gym's
  proposed reference is `fairbatching_pab`. I did not tune the setup toward them. `validate.py`
  passes, including `--post-design`.
- **Hardware speed is calibrated to the paper's regime, not to specs.** A pure bf16 spec estimate
  for H20 (0.25 ms/token) saturates the node at half the load the paper's Figure 5(f) shows; I used
  0.15 ms/token so the load axis matches. With PAB, peak goodput still rises at the top load
  (3.92 RPS at 5 RPS offered), as in the paper's figure at 4 RPS. For admission-control methods,
  "peak" is therefore effectively the goodput at the top load.
- **Not covered:**
  - The paper's cluster experiment (PAB load balancer, Figure 8).
  - The BurstGPT and Azure traces.
  - The other three models.
  - KV-memory pressure.
- **Evaluation wall-time cost of stalling.** A program that sleeps can delay a hidden evaluation by
  up to ~10–20 minutes before it fails. That delay is bounded, but not zero.
- **Compute limits** are on the scheduler's own CPU (300 s per run, 3600 s per evaluation). A
  scheduler that is honest but computationally heavy would fail them. Real step-level schedulers
  must decide in milliseconds, so this is a deliberate constraint.

## 7. Weights

0.5 for single-node peak goodput: the paper's headline "20.0% single-node capacity" and Table 3.
0.25 each for the Table 4 latency detail and the Table 5 SLO-versatility row. The old 0.7 / 0.3
split had pointed at a saturated metric and an experiment that was never emitted.
