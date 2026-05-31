# Overview

FairBatching is an LLM inference scheduling paper. It argues that strict
decode-prioritizing schedulers over-serve decode work, waste accumulated TPOT
slack, and starve prefill bursts, causing poor TTFT tail latency.

This workspace packages a trace-driven reproduction of the scheduler logic and
scores it against prefill-priority and Sarathi-style baselines on request-level
TTFT, TPOT, SLO violation rate, and effective RPS.
