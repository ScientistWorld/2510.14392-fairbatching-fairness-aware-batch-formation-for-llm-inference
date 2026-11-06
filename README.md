# FairBatching: Fairness-Aware Batch Formation for LLM Inference

| This paper | Scientist World, the framework this environment belongs to |
|:---:|:---:|
| <a href="https://arxiv.org/abs/2510.14392"><img alt="This paper on arXiv" src="https://img.shields.io/badge/arXiv-2510.14392-b31b1b?logo=arxiv&logoColor=white"></a> | <a href="https://arxiv.org/abs/2611.08478"><img alt="Scientist World on arXiv" src="https://img.shields.io/badge/arXiv-2611.08478-b31b1b?logo=arxiv&logoColor=white"></a> <a href="https://github.com/zlab-princeton/scientist_world"><img alt="Scientist World code" src="https://img.shields.io/badge/Code-Scientist_World-181717?logo=github&logoColor=white"></a> <a href="https://github.com/zlab-princeton/scientist_world_cli"><img alt="Command line code" src="https://img.shields.io/badge/Code-Command_line-181717?logo=github&logoColor=white"></a> |

| | |
|---|---|
| **Paper ID** | [2510.14392](https://arxiv.org/abs/2510.14392) |

## Summary

FairBatching is an LLM inference scheduling paper. It argues that strict decode-prioritizing schedulers over-serve decode work, waste accumulated TPOT slack, and starve prefill bursts, causing poor TTFT tail latency.

## Run it

```bash
pip install sciworld
sw research 2510.14392 --mode innovate --agent claude-code
sw evaluate 2510.14392
```

## More

The full problem statement, method notes, evaluation protocol and reproduction log are in [DETAILS.md](DETAILS.md).

