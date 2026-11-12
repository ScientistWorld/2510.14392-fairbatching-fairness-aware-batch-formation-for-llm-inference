# Problem

An LLM inference engine serving many concurrent requests runs in discrete
steps.  At every step its scheduler forms one batch that mixes two kinds of
work:

- **prefill** of newly arrived prompts (a whole prompt or a chunk of it); a
  request emits its first output token when its prefill completes;
- **decode** of requests that are already generating: one new token per
  request per step.

The duration of a step grows with the number of new tokens in the batch and
with the context (KV cache) the batch touches.  Every request carries two
latency objectives:

- **TTFT** (time to first token): the first token must be produced within
  `ttft_slo_ms` of arrival;
- **TPOT** (time per output token): output token `j` (0-based, the first token
  is `j = 0`) must be produced by `arrival + ttft_slo_ms + tpot_slo_ms * j`.

A request is served successfully only if it meets both.  The scheduler may
also refuse (reject) a request before its prefill starts — for example to
hand it to another node — but a rejected request counts as a failure.

The task: given online arrivals (the scheduler never sees the future and never
learns a request's output length in advance), choose admissions and batches
so that as many requests as possible meet their objectives, across loads from
light to heavy overload, while keeping first-token and per-token latency tails
low.  The headline quantity is **goodput** (effective requests per second):
offered request rate times the fraction of requests that meet both objectives.

The workloads are real chat traffic (the head of the Qwen-Bailian To-C
trace: prompt avg ~900 tokens, output avg ~380 tokens, bursty arrivals)
replayed at controlled request rates on one simulated Qwen3-14B / H20-class
GPU node.
