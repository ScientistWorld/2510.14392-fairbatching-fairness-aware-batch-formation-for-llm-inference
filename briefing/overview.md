# Overview

FairBatching is an LLM inference scheduling paper.  It argues that
decode-prioritizing stall-free schedulers (Sarathi) over-serve decode work,
waste accumulated TPOT slack and starve prefill bursts, while prefill-first
schedulers (vLLM default) stall decodes.  It proposes per-token envelope
deadlines, an adaptive time budget, three-stage batch formation and a prefill
admission budget (PAB) for load control.

This gym runs schedulers as programs against an evaluator-owned simulator of a
Qwen3-14B / H20-class node, on workloads sampled from the paper's own
QwenTrace (Qwen-Bailian trace A head segment).  It scores single-node peak
goodput over a load sweep (Table 3 / Figure 5), latency tails at the Table 4
operating point, and peak goodput under the relaxed TTFT 1000 / TPOT 100 ms
SLOs of Table 5.  Held-out workloads are generated only inside `eval/test/`.
