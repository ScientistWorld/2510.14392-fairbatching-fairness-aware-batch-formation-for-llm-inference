"""Offline profiling: fit the linear step-time model to the simulated GPU.

Mirrors the paper's offline profiling step (Section 3.2/4): sample random
mixed prefill/decode batches, time them on the "hardware", and fit
``a + b * new_tokens + c * total_context`` by least squares.  The fitted
values are published to schedulers as ``PROFILE`` in ``data/serving_sim.py``.
"""

from __future__ import annotations

import random
from pathlib import Path

from data.serving_sim import HARDWARE, load_pool


def sample_batches(pool, rng, n_batches=4000):
    rows = []
    for _ in range(n_batches):
        n_dec = rng.randint(0, 96)
        n_pre = rng.choice([0, 0, 1, 1, 2, 3])
        tokens = 0
        ctx = 0
        attn = 0.0
        for _ in range(n_dec):
            _, p, o = pool[rng.randrange(len(pool))]
            g = rng.randint(1, max(1, o))
            tokens += 1
            ctx += p + g
        for _ in range(n_pre):
            _, p, _ = pool[rng.randrange(len(pool))]
            done = rng.choice([0, 0, rng.randrange(0, p)]) if p > 1 else 0
            chunk = rng.randint(1, max(1, min(p - done, 4096)))
            if tokens + chunk > HARDWARE["max_batch_tokens"]:
                continue
            tokens += chunk
            ctx += done
            attn += chunk * (done + chunk / 2.0)
        if tokens == 0:
            continue
        eps = rng.uniform(-HARDWARE["noise"], HARDWARE["noise"])
        t = (HARDWARE["a"] + HARDWARE["b"] * tokens + HARDWARE["c"] * ctx + HARDWARE["d"] * attn) * (1 + eps)
        rows.append((tokens, ctx, t))
    return rows


def fit(rows):
    # normal equations for [1, tokens, ctx]
    s = [[0.0] * 3 for _ in range(3)]
    v = [0.0] * 3
    for tokens, ctx, t in rows:
        x = (1.0, float(tokens), float(ctx))
        for i in range(3):
            v[i] += x[i] * t
            for j in range(3):
                s[i][j] += x[i] * x[j]
    # gaussian elimination
    m = [row[:] + [v[i]] for i, row in enumerate(s)]
    for col in range(3):
        piv = max(range(col, 3), key=lambda r: abs(m[r][col]))
        m[col], m[piv] = m[piv], m[col]
        for r in range(3):
            if r != col:
                f = m[r][col] / m[col][col]
                for k in range(col, 4):
                    m[r][k] -= f * m[col][k]
    return [m[i][3] / m[i][i] for i in range(3)]


def main() -> None:
    pool = load_pool(Path("data/qwen_trace_head.csv"))
    rng = random.Random(20251016)
    rows = sample_batches(pool, rng)
    a, b, c = fit(rows)
    errs = [abs((a + b * tok + c * ctx) - t) / t for tok, ctx, t in rows]
    errs.sort()
    print(f"fit: a={a:.4f} b={b:.6f} c={c:.3e}")
    print(f"relative error: median {errs[len(errs)//2]*100:.2f}%  p95 {errs[int(.95*len(errs))]*100:.2f}%")


if __name__ == "__main__":
    main()
