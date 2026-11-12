# Progress

Current state (iteration 11): see `HARDENING.md` for the full account.

- `method/fairbatching.py`: FairBatching (envelope deadlines, adaptive time
  budget, Algorithm 1 batch formation) and FairBatching-PAB (Appendix A
  admission budget) as protocol scheduler programs.
- `baseline/schedulers.py`: vLLM-vanilla and tuned vLLM-sarathi.
- `data/`: QwenTrace head segment (`qwen_trace_head.csv`, from
  `scripts/download.sh`), public simulator `serving_sim.py`, protocol helper
  `sched_client.py`, profiling script.
- `eval/train/`, `eval/test/`: evaluators (test is self-contained, undisclosed
  seeds).
- `scripts/run.sh` reproduces the reference end to end; `python
  redteam/hardening/run_audit.py` reruns the hardening audit.
- `python validate.py` passes.

Earlier iterations (synthetic traces, artifact-based scoring) are described in
`DETAILS.md` and superseded.
