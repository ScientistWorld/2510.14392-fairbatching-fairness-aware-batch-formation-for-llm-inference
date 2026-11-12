# Audit progress

2026-11-12: Two CPU jobs evaluated real stronger/weaker scheduling
variants, three deterministic reference seeds, a 20-configuration simple search,
and artifact/coverage shortcuts. Results and explicit-cohort ranks are recorded
in `analysis.json` and `../REDTEAM.md`. A final CPU job repeats the two search
cases with a selection rule that uses only public train objectives, removing
the audit harness's earlier dependency on unavailable metric coefficients.
The audit makes no new reproduction
milestone claim and does not modify the gym's existing `MILESTONES.md`.
