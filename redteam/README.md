This directory contains benchmark audit evidence. Run the complete suite as a
CPU job with `python -m redteam.run_all`. Each named attempt also has a `run.sh`
which resets the audit fixture, installs its method, invokes public train
evaluation, and runs `bash scripts/evaluate_test.sh` with no arguments in a new
process. `run.json` preserves the actual hidden-test exit code. Invalid attempts
are not counted as successful fresh scores.

`support/quality_candidate.py` changes real scheduling while retaining execution
costs and the complete trace. `support/shortcut_candidate.py` contains explicitly
named artifact shortcuts. Neither imports the hidden evaluator, uses the paper,
or changes the evaluator's Python environment. The audit driver invokes the
test script only to measure the results and save evidence.

The cached original timing artifacts are reusable earned reference outputs,
not hidden answers. They can be regenerated with the visible baseline and
method. The cached-answer shortcut deliberately shows that this public, fixed
trace can be memorized and replayed without running a scheduler at evaluation.

The report's numerical ranks use absolute coefficients, the specified better
directions, experiment weights, and population standard deviations over its
explicit audit cohort. Production ranks depend on the unknown future cohort;
Pareto improvements and exact ties are independent of that choice.
