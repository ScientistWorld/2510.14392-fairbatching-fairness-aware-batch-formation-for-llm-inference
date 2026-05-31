# Constraints

Future methods must use the same trace input format and must write scheduler
outputs with request-level timings to `outputs/*.json`.

Do not improve scores by changing the workload, shortening requests, relaxing
SLOs, or excluding difficult requests. Rejected requests count as SLO failures
for `slo_violation_rate`.

The evaluator must remain method-agnostic and must not import from `method/`.
