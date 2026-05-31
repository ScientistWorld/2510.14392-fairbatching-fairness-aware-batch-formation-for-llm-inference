# Problem

LLM serving systems must schedule a mix of prefill work for newly arrived
requests and decode work for active generations. The problem is to maximize
the number of requests that satisfy both latency objectives:

- TTFT: time from request arrival to first output token.
- TPOT: worst average time per output token after the first token.

The scheduler receives a trace of request arrivals, prompt lengths, output
lengths, and per-request SLOs. It emits request-level completion timings.
Success requires high effective requests per second while keeping both TTFT and
TPOT violations low.
