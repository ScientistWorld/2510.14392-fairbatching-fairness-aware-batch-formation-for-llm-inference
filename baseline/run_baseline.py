"""Run baseline schedulers using method-agnostic simulator utilities."""

from __future__ import annotations

import argparse
from pathlib import Path

from data.serving_sim import (
    BatchItem,
    ModelParams,
    Request,
    all_waiting_prefill,
    one_decode,
    run_simulation,
    token_time,
)


class SarathiScheduler:
    """Decode-prioritizing stall-free batching baseline."""

    def __init__(self, params: ModelParams):
        self.params = params

    def select(self, active: list[Request], now_ms: float) -> list[BatchItem]:
        token_budget = self.params.sarathi_token_budget
        remaining_time = min((r.tpot_slo_ms for r in active if r.is_decode), default=50.0) - self.params.a
        batch: list[BatchItem] = []
        for req in sorted([r for r in active if r.is_decode], key=lambda r: r.request_id):
            item = one_decode(req)
            if token_budget >= 1 and token_time(item, self.params) <= remaining_time:
                batch.append(item)
                token_budget -= 1
                remaining_time -= token_time(item, self.params)
        for req in sorted([r for r in active if r.is_prefill], key=lambda r: r.arrival_ms):
            if token_budget <= 0:
                break
            chunk = min(req.prompt_len - req.prefill_done, token_budget)
            if self.params.b * chunk + self.params.c * req.context_len > remaining_time:
                chunk = int(max(0, (remaining_time - self.params.c * req.context_len) / self.params.b))
            if chunk > 0:
                batch.append(BatchItem(req, chunk, "prefill"))
                token_budget -= chunk
                remaining_time -= self.params.b * chunk + self.params.c * req.context_len
        return batch


class VanillaScheduler:
    """Prefill-prioritizing continuous batching baseline."""

    def __init__(self, params: ModelParams):
        self.params = params

    def select(self, active: list[Request], now_ms: float) -> list[BatchItem]:
        token_budget = self.params.max_token_budget
        batch: list[BatchItem] = []
        for req in sorted([r for r in active if r.is_prefill], key=lambda r: r.arrival_ms):
            if token_budget <= 0:
                break
            item = all_waiting_prefill(req)
            chunk = min(item.new_tokens, token_budget)
            batch.append(BatchItem(req, chunk, "prefill"))
            token_budget -= chunk
        for req in sorted([r for r in active if r.is_decode], key=lambda r: r.request_id):
            if token_budget <= 0:
                break
            batch.append(one_decode(req))
            token_budget -= 1
        return batch


SCHEDULERS = {
    "vllm_sarathi": SarathiScheduler,
    "vllm_vanilla": VanillaScheduler,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, default=Path("data/qwen_synthetic.csv"))
    parser.add_argument("--policy", choices=sorted(SCHEDULERS), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    params = ModelParams()
    run_simulation(args.trace, args.policy, SCHEDULERS[args.policy](params), args.output, params)


if __name__ == "__main__":
    main()
