# Experiments

All experiments run every registered scheduler online against the same
evaluator-owned workload instances (paired comparison), sampled from real
chat traffic (QwenTrace head segment) on one simulated Qwen3-14B / H20-class
node.  Held-out instances come from undisclosed seeds.

## qwen_peak_goodput (weight 0.5)

Establishes how much load a single node can carry while keeping requests
within a 500 ms first-token / 50 ms per-token objective.  The offered rate is
swept from light load (1 RPS) to heavy overload (5 RPS); goodput (requests per
second that meet both objectives, refusals counting as failures) is measured
at each load.  The peak of that curve is the node's usable capacity; the mean
over the sweep additionally rewards degrading gracefully once the node is
overloaded.  Both are expressed relative to the better of two standard
scheduling baselines (prefill-first and decode-first stall-free batching).

## qwen_latency_detail (weight 0.25)

Establishes whether a scheduler that serves more requests within the
objectives also keeps latency tails low at a fixed, busy operating point
(2.5 RPS, near the baselines' capacity): first-token latency (median and
P99), the P99 per-token output rate, and the fraction of all requests served
within both objectives.  The latency numbers only count when the scheduler
keeps at least as many requests within the objectives as the weaker
baseline, so refusing hard requests cannot buy good tails.

## qwen_relaxed_slo_goodput (weight 0.25)

Establishes whether the capacity gain persists when the objectives are
relaxed to 1000 ms first-token / 100 ms per-token, where simple schedulers
already meet the objectives more often and the room for improvement changes.
Same goodput measurement as `qwen_peak_goodput`, over 1.5–6 RPS.
