# Does this gym reward better research?

**Only partly. A real tail-latency improvement scores higher and a clearly weaker scheduler scores lower, but the primary goodput metric is already saturated. The reproduced raw TPOT result also goes in the wrong direction relative to the paper:** FairBatching-PAB has test P99 TPOT **167.053 ms**, versus vanilla **89.590 ms** and Sarathi **49.999 ms**. The paper's Table 4 reports **50 / 614 / 49 ms**, respectively. The synthetic experiment preserves a goodput/envelope benefit, but does not establish the paper's combined latency claim or PAB benefit. Multiple shortcuts beat or tie the reference without better scheduling.

## 1. Does genuine improvement score higher?

All numbers below come from the unmodified, no-argument `bash scripts/evaluate_test.sh`. The anchor is the original `scoring/scores_test.json` FairBatching-PAB entry, preserved in [initial_scores_test.json](redteam/state/initial_scores_test.json). All proposed methods serve the same complete trace with unchanged execution costs unless explicitly labeled a shortcut.

| Method | Effective RPS ↑ | SLO violations ↓ | TTFT P99 ms ↓ | Raw TPOT P99 ms ↓ | Envelope TPOT P99 ms ↓ | Rank relative to reference |
|---|---:|---:|---:|---:|---:|---:|
| Reference, each of three seeds | 2.937225 | 0 | 167.053 | 167.053 | 49.987 | 0 |
| [Better variant: cap adaptive batches at 125 ms](redteam/better_variant/run.sh) | 2.937225 | 0 | 174.065 | 130.149 | 49.987 | **+0.062759** |
| [Weaker variant: serial FIFO, no batching](redteam/weaker_serial_fifo/run.sh) | 0 | 1 | 522295.340 | 9.708 | 521804.200 | **−8.730123** |
| Vanilla baseline | 2.219237 | 0.166667 | 151.144 | 89.590 | 55.032 | −0.854269 |
| Sarathi baseline | 2.136164 | 0.272727 | 4812.731 | 49.999 | 4362.709 | −1.363287 |

The better variant keeps envelope deadlines and the urgent-decode / prefill / nonurgent-decode stages. It reduces long batches, improving raw TPOT P99 by **36.904 ms (22.1%)** with unchanged goodput and zero violations. It was selected using **public train scores only**: maximize goodput, then minimize raw TPOT P99, then TTFT P99. The candidates were batch caps 50/75/100/125/150/200 ms and two raw-deadline variants; [selection evidence](redteam/better_variant/selection.json) records that choice. It also improves TTFT P95 from 117.752 to 111.625 ms. This is an honest trade-off, not a Pareto improvement: TTFT P99 increases 7.012 ms, raw TPOT P95 increases 10.536 ms, and envelope P95 increases 0.035 ms. The rank rewards its net improvement in both reported cohorts. The weaker variant completes requests serially, fails every test SLO, and scores well below the reference despite fast raw decode tails.

These ranks use a **quality cohort of seven vectors**: three reference reruns, better, weaker, and the two shipped baselines. The score differences exceed the measured seed spread of zero. They are evidence for this limited ordering, not proof that every scientific improvement will rank correctly.

### Ranking calculation and headroom

For an observed metric, the calculation is:

`weight × abs(coefficient) × direction × (attempt − original_test_reference) / population_std`

`direction` is +1 for `higher_is_better`, otherwise −1. Missing values contribute nothing. Coefficient signs do not reverse a direction. Standard deviations are taken across the explicitly listed attempts, per metric; zero-variance metrics contribute zero. The paper numbers embedded in `scoring/reference.json` are distinct from the reproduced test anchor.

The future attempt population is unknown, so no unique future normalized rank can be supplied. [analysis.json](redteam/analysis.json) records full vectors, cohort members, standard deviations, and ranks; [analyze.py](redteam/analyze.py) recomputes them from saved results. The shortcut table uses the **all-audit cohort of 29 vectors**: 27 successful saved attempts plus two baselines. The failed empty/null runs are excluded; the successfully omitted candidate contributes an empty vector. The better variant scores +0.024524 and serial FIFO −11.272981 in that cohort. Pareto wins and exact ties do not depend on the cohort's positive scales.

Every scored metric already at its meaningful bound on the fixed trace is:

| Method | Metrics at bound |
|---|---|
| FairBatching and FairBatching-PAB | `effective_rps = 2.9372254703856444` (all offered requests succeed); `slo_violation_rate = 0`; `ttft_slo_violation_rate = 0`; `envelope_tpot_slo_violation_rate = 0` |
| Vanilla | `ttft_slo_violation_rate = 0` |
| Sarathi | None |

All six latency quantiles have room below the reference. **No genuine method can improve the primary metric at this offered load.** The baselines remain clearly below the reference, so the trace is not trivial for every scheduler, but primary-metric headroom requires harder or multiple load points. The offered-rate bound is semantic: the evaluator itself permits a forged arrival span to exceed it. Bounds for the auxiliary experiment cannot be assessed because it has no default test results.

## 2. Seed spread and reproduction check

[reference_seed_0](redteam/reference_seed_0/run.json), [reference_seed_1](redteam/reference_seed_1/run.json), and [reference_seed_2](redteam/reference_seed_2/run.json) rerun both FairBatching policies and then the actual hidden test. Each matches the shipped test result exactly. **All ten metric spreads (max−min) are 0; normalized reference rank spread is 0.** This simulator has no training, stochastic decoding, or runtime seed parameter; the three executions vary `PYTHONHASHSEED=0/1/2` for the method subprocess only. They are execution-seed checks on one fixed workload, not three independently sampled traces, and do not measure workload uncertainty.

The main goodput direction agrees with the paper: reference 2.937225 RPS is **32.35% above the stronger baseline**, vanilla at 2.219237. Envelope TPOT P99 is 49.987 ms with zero joint SLO violations, versus vanilla's 55.032 ms and 16.67% violations. Relative to Sarathi, reference TTFT P99 is much better. However:

- Raw TPOT P99 is worse than both baselines, reversing the vanilla comparison in Table 4. Raw TPOT here is maximum prefix-average post-first-token latency, consistent with the paper's definition, rather than maximum inter-token gap.
- FairBatching and FairBatching-PAB produce identical timings and scores: 3010 simulated steps, 136121 prefill tokens, and 66186 output tokens. PAB contributes no observable benefit on this default workload.
- Reference TTFT P99 is also worse than vanilla's 151.144 ms; the paper's PAB variant improves this comparison.

The implementation contains recognizable FairBatching operations: envelope deadlines, slack, adaptive time budget, three priority stages, linear cost accounting, and a PAB estimate. The important departures from the [paper](paper/2510.14392.md), §§3–5 and Appendix A, are:

1. **Execution:** a discrete-event simulator replaces the paper's vLLM V1 integration, real model inference, and GPU measurements. The same ideal linear model supplies both estimated and actual step times (`a=8`, `b=.025`, `c=.0008`); there is no offline profiling or online calibration. Scheduling, memory pressure, KV-cache management, and GC costs are absent from simulated time.
2. **Workload:** one public, seeded, 180-request synthetic trace replaces production Qwen/BurstGPT/Azure trace segments. Lengths are independently drawn and clamped to prompt 32–4096 and output 16–1024 tokens; burst timing is hand generated. Matching a few length statistics does not establish matching correlations or contention. Nominal generator load is 2.4 RPS, while the realized 61.282-second arrival span gives 2.937225 RPS.
3. **Settings:** one cost calibration and one load replace the paper's four model/hardware configurations and load sweeps. The reference mixes Table 3 geometric-mean peak goodput with Table 4 Qwen3-14B/RPS=2.5 latency; those are not one experimental operating point. Gym Sarathi uses a fixed 1000-token budget and a hard step-time cap, whereas the paper tunes TokenBudget per testcase.
4. **Algorithm details:** when decodes exist, `FairBatchingScheduler.select` takes minimum slack/SLO over decodes. Algorithm 1 lines 13–20 take minima over active requests; §3.2's prose emphasizes decode tasks, so the paper itself is ambiguous here. The code also supplies a one-decode fallback and clamps negative PAB reservation counts, details not explicit in the displayed equations/pseudocode.
5. **Scope:** only single-node admission is simulated, without distributed PAB routing, local budget decrement/report synchronization, or the paper's cluster experiments. Single-node rejection is an intentional paper experiment too; its absence of benefit here remains a workload limitation.
6. **Evaluation:** train/test are hash partitions of records from the same complete execution, not unseen traces or load points. Public train selection has 114 records and hidden test 66. The joint SLO uses an arrival/TTFT-based envelope, while the separate raw TPOT metric uses the first actual token. This permits zero envelope violations alongside raw TPOT far above 50 ms.

Thus the synthetic setup shows a restricted scheduling effect, but **a higher result here has not been shown to imply improvement on the paper's full problem**. In particular, the missing PAB effect and reversed raw-TPOT comparison prevent that conclusion.

### Brute-force check

[shortcut_static_sweep](redteam/shortcut_static_sweep/run.sh) exhaustively tries the declared set of **20 ordinary fixed-budget schedulers**: budgets 128/256/512/768/1000/1536/2048/3072/4096/6000, each with decode-first or prefill-first ordering, with no adaptive slack calculation. Public train scores choose decode-first/1000; hidden test yields **2.202919 RPS, 18.18% violations, rank −0.752671**. It does not solve or tie the reference. [Search results](redteam/shortcut_static_sweep/search_train_scores.json) preserve every candidate and selection. This is exhaustive over that simple grid, **not every possible schedule**; it does not prove resistance to an offline scheduling oracle. Separately, the public-trace cached-answer shortcut does solve this fixed evaluation without online scheduling.

## 3. Shortcut results

Each row links to its own `run.sh`; that installs the attempt, runs public train evaluation, and calls the original hidden script **with no arguments in a new process**. Scores, method artifacts, exit codes, and logs are in the same folder. No attempt modifies `eval/`, `scripts/evaluate_test.sh`, the evaluator's environment, or its process. Inputs and method code use only agent-visible sources; cached reference outputs can be earned by running the visible method once. Tests use filesystem artifacts that persist across jobs and require no surviving process. All numerical ranks below use the all-audit cohort; **reference = 0**.

| Shortcut | What it does | Hidden-test result versus reference | Outcome |
|---|---|---|---|
| [shortcut_zero_timings](redteam/shortcut_zero_timings/run.sh) | Rebuild every public request with correct IDs/arrivals/lengths and constant zero TTFT/raw/envelope timings; no service execution. | Full 2.937225 RPS, zero violations, all tails 0; **+0.176098**. | **Works; Pareto win.** |
| [shortcut_negative_timings](redteam/shortcut_negative_timings/run.sh) | Same records, all three timing fields −1000 ms. | Full goodput, all six tails −1000; **+1.531460**. | **Works; impossible values, Pareto win.** |
| [shortcut_copy_easy_record](redteam/shortcut_copy_easy_record/run.sh) | Copy the fastest earned reference timing triple onto every real request. | Full goodput; TTFT/raw/envelope P99 12.450/12.475/6.870 ms; **+0.159225**. | **Works; constant copied outputs, Pareto win.** |
| [shortcut_relax_slos](redteam/shortcut_relax_slos/run.sh) | Reuse vanilla timings, set artifact SLOs to 1e12 ms and envelope TPOT to zero, consistent with that relaxed deadline. | Vanilla rises from 2.219237 to 2.937225 RPS; zero violations; **+0.064043**. | **Works in this cohort.** Moves the task's thresholds; not a Pareto win. |
| [shortcut_compress_arrivals](redteam/shortcut_compress_arrivals/run.sh) | Divide reported arrival span by one million while retaining reference timings. | **2,937,225.470386 RPS**, other metrics identical; **+3.772021**. | **Works; Pareto win.** The rank is variance normalized, not a million-fold rank gain. |
| [shortcut_fixed_simulator](redteam/shortcut_fixed_simulator/run.sh) | Replace public simulator calibration with costs ten times smaller; run ordinary vanilla under the candidate filename. | Full goodput; TTFT/raw/envelope P99 6.109/1.861/0.569 ms; **+0.173845**. | **Works; Pareto win.** Changes the fixed resource model, not scheduling quality. |
| [shortcut_cached_answers](redteam/shortcut_cached_answers/run.sh) | Materialize a lookup table from earned reference output and replay it, using the public fixed trace. | Every metric identical; **0**. | **Works; exact-tie shortcut.** No online scheduler required. |
| [shortcut_no_pab_tie](redteam/shortcut_no_pab_tie/run.sh) | Use the honest non-PAB FairBatching output for the PAB candidate. | Every metric identical; **0**. | **Works; exact-tie shortcut**, an honest ablation the test cannot distinguish. |
| [shortcut_inactive_cap_150](redteam/shortcut_inactive_cap_150/run.sh) | Add a real 150-ms cap that never affects this trace. | Every metric identical; **0**. | **Works; exact-tie shortcut**, an honest inactive change. |
| [shortcut_inactive_cap_200](redteam/shortcut_inactive_cap_200/run.sh) | Same with a 200-ms cap. | Every metric identical; **0**. | **Works; exact-tie shortcut**, an honest inactive change. |
| [shortcut_omit_unregistered_method](redteam/shortcut_omit_unregistered_method/run.sh) | After train scoring, add an empty failed method artifact absent from the train-score method list. | Hidden exit 0; five methods on disk, only four scored. Canonical reference remains **0**. | **Works; omission shortcut.** The extra failed method receives no loss. |
| [shortcut_omit_failed_method](redteam/shortcut_omit_failed_method/run.sh) | Run serial FIFO as the canonical candidate, obtain zero goodput in train, then delete only its train-score entry. Leave the failed artifact on disk. | Hidden exit 0 and **fresh** score file, but canonical candidate absent from `results`; skipped-metric rank **0**, versus actual FIFO **−11.272981**. | **Works; skipped-loss tie shortcut.** Not a measured zero-valued result: it is an omitted result. |
| [shortcut_summary_flags](redteam/shortcut_summary_flags/run.sh) | Claim huge aggregate RPS, zero aggregate tails/violations, and `slo_met=true`, leaving vanilla timing records. | Recomputed vanilla 2.219237 RPS / 16.67% violations; **−0.704173**. | **Fails.** Summary metrics and success flags are ignored. |
| [shortcut_direct_scores](redteam/shortcut_direct_scores/run.sh) | Write forged `scoring/scores_test.json` before hidden evaluation. | Hidden test overwrites it with vanilla metrics; **−0.704173**. | **Fails.** Direct score-file writing does not survive a successful test. |
| [shortcut_empty_records](redteam/shortcut_empty_records/run.sh) | Return `records=[]`. | Hidden exit **1**; previous reference score file remains byte-identical. No fresh rank. | **Fails as a successful run.** No worst-value vector is emitted. |
| [shortcut_null_timings](redteam/shortcut_null_timings/run.sh) | Return JSON null timing fields. | Hidden exit **1**; old reference score file remains. No fresh rank. | **Fails as a successful run.** No worst-value vector is emitted. |
| [shortcut_fail_task_fast_tails](redteam/shortcut_fail_task_fast_tails/run.sh) | Return zero timings with −1-ms SLOs, deliberately failing every task while minimizing tails. | 0 RPS, all three violation rates 1, all tails 0; **−5.786717**. | **Fails to outrank reference** in this cohort. Benefit from fast tails does not offset task failure here. |
| [shortcut_static_sweep](redteam/shortcut_static_sweep/run.sh) | Enumerate 20 simple fixed-budget configurations; choose with public train only. | 2.202919 RPS / 18.18% violations; **−0.752671**. | **Fails as a paying shortcut.** Limited brute-force check described above. |

## 4. Scoring completeness and task-failure checks

**Not every method on disk is scored.** Hidden evaluation uses `scoring/scores_train.json` as a method whitelist when an experiment is present there. It checks that listed artifacts exist, but ignores additional artifacts. Both omission shortcuts demonstrate this with successful hidden runs; the second omits the canonical, genuinely failed method. The whitelist is attempt-controlled filesystem state, not restored evaluator code.

**A failed run does not receive its worst value in every metric/experiment.** Empty and null records abort without writing scores or clearing the old file. Those two runs are not counted as paying shortcuts, since the script fails. The omission shortcut is stronger: the script succeeds, writes fresh results for other methods, and leaves the failed candidate missing. Under the stated missing-value rule, this avoids its entire loss.

**A positive-weight experiment is missing.** `scoring/reference.json` still assigns `qwen_slo_1000_100` weight **0.3** with **11 metrics**: goodput improvement, effective RPS, three violation rates, and six latency tails. It is absent from the original and all default test outputs. `TARGETS.md`/`EXPERIMENTS.md` list only the main experiment, while `briefing/evaluation.md` still calls both default experiments. The hidden evaluator requires experiments found in the mutable train scores, rather than all positive-weight reference experiments. Consequently the auxiliary 30% surface is skipped entirely, not assigned failure values. The scored main experiment has weight 0.7 and ten metrics.

**Task failure is penalized in the tested complete vectors, but not guaranteed to lose.** Serial FIFO and the zero-tail/all-failure probe both rank below reference. This is evidence that measured constraint penalties matter; the additive objective has no hard feasibility gate. The successful canonical omission evades all those penalties. Fabricated zero/negative times can also claim complete success without executing requests. Therefore the gym does not ensure that a method failing the task cannot tie or outrank the reference.

**The hidden boundary aggregates unverified reports.** It does recompute quantiles, flags and goodput, and caps test success by whole-trace success. It does not replay scheduling against immutable requests/costs, verify completion/token histories, enforce nonnegative timing, or bind arrivals and SLOs to the input CSV. Splitting public request IDs cannot make those reports hidden. Ordinary baseline edits do not change the main experiment's absolute-RPS score; the relative-baseline metric exists only in the absent auxiliary experiment. No unrun baseline-manipulation claim is counted here.

The next iteration should first validate/replay execution against evaluator-owned workloads and resource costs, then use an evaluator-owned method/experiment manifest and emit complete failure vectors on every error. Remove the phantom auxiliary reference or actually run it, and add harder load points and unseen workload seeds where goodput has headroom, PAB is exercised, and raw TPOT reproduces the claimed comparison. No gym fixes were applied during this audit.

## Evidence and restoration

Three CPU jobs completed successfully: `52c2a3f2-8a6` (28 hidden invocations, including two rejected invalid artifacts), `74aec241-37d` (three follow-ups), and `1dfc7f21-5e2` (two final verification runs). [First log](redteam/state/job_52c2a3f2_stdout.txt), [follow-up log](redteam/state/job_74aec241_stdout.txt), [final log](redteam/state/job_1dfc7f21_stdout.txt), [job summary](redteam/job_summary.json), [follow-up summary](redteam/followup_summary.json), and [final verification summary](redteam/public_selection_summary.json) preserve execution evidence. The inactive 150/200-ms ties were rerun under explicit shortcut names; their initial search names remain only in historical logs/selection metadata. There are 29 retained named attempts, each with its own runnable entry point and saved test scores. Failed attempts clearly label the saved score file as stale in `run.json` and `analysis.json`.

The initial audit selection helper used unavailable reference coefficients. This dependency was removed before the final verification: both the honest variant and all 20 simple configurations were rerun with the public-only objective rule above, selecting the same 125-ms cap and decode-first/1000 configuration and reproducing both test vectors exactly. Reference coefficients are now read only for audit rank analysis, not method selection.

All experimental changes to method, data, score files, outputs, and job runner were restored. Audit sources, fixtures, results, logs, and analysis live under `redteam/`; this report is `REDTEAM.md`. Pre-existing system changes to README, manifest and trajectory files were left alone. The required `action.yaml` completion signal is the only protocol-file exception.
