# Constraints

Future methods must use the same trace input format and must write scheduler
outputs with request-level timings to `outputs/*.json`.

Do not improve scores by changing the workload, shortening requests, relaxing
SLOs, or excluding difficult requests. Rejected requests count as SLO failures
for `slo_violation_rate`.

Raw `tpot_p99_ms` is a paper-style latency constraint, while
`envelope_tpot_p99_ms` and `envelope_tpot_slo_violation_rate` measure the
deadline-envelope SLO accounting used by the evaluator. Improvements should
preserve both views unless the change is explicitly documented as a metric
trade-off.

The evaluator must remain method-agnostic and must not import from `method/`.
