# Method

FairBatching replaces static token-budget scheduling with SLO-aware batch
formation.

The implementation in `method/simulator.py` follows the paper's computation:

- envelope-line deadlines for each request's next token;
- slack calculation from current time to the next deadline;
- adaptive time budget based on decode-request slack and TPOT SLO;
- linear batch-time estimator `a + b * total_new_tokens + c * total_context`;
- three-stage packing of urgent decode tasks, prefill tasks, and non-urgent
  decode tasks;
- the PAB admission-control estimate from Appendix A for the PAB variant.

Baselines implement prefill-prioritizing continuous batching and Sarathi-style
decode-prioritizing stall-free batching.
