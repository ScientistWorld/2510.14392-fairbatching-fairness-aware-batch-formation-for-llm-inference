# Direction

Work within the paper's problem space: step-level scheduling of colocated
prefill and decode work on one LLM inference node under per-request TTFT and
TPOT objectives, with optional admission control.

Valid directions include how batch capacity is chosen, how work is ordered
inside a batch, step-time estimation (the published linear profile is only an
approximation of the measured step durations), admission and load control,
and robustness across load levels and SLO settings.

Out of scope: changing the model, hardware, workload, SLOs or load points;
reading evaluator internals or future requests; optimizing a different
latency objective.
