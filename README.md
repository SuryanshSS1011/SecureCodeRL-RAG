# CARGO: Continuous Retrieval-Grounded Reward Design for Secure Code Generation on Small Language Models

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23008092.svg)](https://doi.org/10.5281/zenodo.23008092)

Code for the ICTAI 2026 paper *Continuous Retrieval-Grounded Reward Design for Secure Code Generation on Small Language Models (CARGO)* by Suryansh Singh Sijwali, Medhansh Kumar Singla, and Suman Saha (Pennsylvania State University).

## Summary

Reinforcement learning with static application security testing (SAST) feedback stalls on small language models: most early rollouts fail to parse, every rollout in a group scores the same near-zero reward, and group-relative or ranking-based optimizers (GRPO, RLOO, RAFT) receive almost no gradient. On Qwen2.5-Coder-1.5B, SAST-only GRPO has a nonzero policy loss on fewer than 9% of training steps.

CARGO (Continuous Augmented Retrieval-Grounded Objective) is a reward-design recipe with three components:

1. **Retrieval-grounded reward R_RAG.** It scores a completion against a secure-fix exemplar retrieved for the prompt's CWE (hybrid BM25 + bge-base retriever), with a 0.95 copy guard. This restores within-group reward variance without replacing the SAST objective.
2. **CWE-aware per-prompt gradient reweighting.** It compensates for supply imbalance across the trained CWEs.
3. **σ-floor on the group-relative normalizer.** It bounds the low-variance bias of dividing by the group standard deviation.

On a 1,582-prompt benchmark covering 19 CWEs in Python, C, and C++, CARGO improves on the SAST-only GRPO baseline by +19.9 pp Compile@1, +16.6 pp Secure@1|Compile, and +26.5 pp Functional-Secure@1. Adding R_RAG improves all four algorithms (GRPO, PPO, RLOO, RAFT), and the gains reproduce on Qwen2.5-Coder-3B and StarCoder2-3B.

## Repository layout

```
src/secure_code_rl_ictai/
├── data_prep/   source adapters (CVEfixes, DiverseVul, Juliet 1.3, CyberSecEval, SecCodePLT,
│                CASTLE, SecurityEval, CWEval), prompt normalization, disjointness audit
├── sast/        CodeQL / Semgrep / Bandit / Cppcheck adapters, SARIF normalizer, CWE hierarchy, CVSS severity
├── reward/      reliability oracle (parse / compile / run / test), reward calculator, reward pipeline
├── rag/         exemplar index, hybrid BM25 + dense retriever, R_RAG
├── rl/          GRPO, PPO (value head), RLOO, RAFT, CWE reweighter, LoRA policy, trainer, step metrics
└── eval/        evaluation harness, metric registry (Compile@1, Secure@1|Compile, Func-Sec@1), baselines
scripts/         corpus build, training, evaluation, rescoring, analysis, SLURM launchers
tests/           unit tests (real_* markers gate tests that need SAST tools, GPUs, or model downloads)
data/            small static artifacts: CWE list, CWE hierarchy, NVD per-CWE CVSS medians
```

## Installation

```bash
python3.11 -m venv .venv && . .venv/bin/activate
pip install -e .
pip install -r requirements.txt -r requirements-eval.txt
```

Some tools are not on PyPI and are installed separately: CodeQL CLI 2.25.4, Cppcheck 2.18.0, and gcc/g++ (the paper runs used gcc 8.5.0). The resolver finds them on `PATH`, or through `CODEQL_BINARY` and `CPPCHECK_BINARY`. Scratch directories can be moved with `ICTAI_ORACLE_TMP` and `ICTAI_PIPELINE_TMP`, and `SLURM_TMPDIR` is honored when it is set.

Hugging Face downloads read `HF_TOKEN`, and NVD lookups read `NVD_API_KEY`, both from `.env` (gitignored).

## Tests

```bash
make test        # fast unit suite
make test-real   # also exec generated code through the reliability oracle
make test-all    # also shell out to the SAST tools
make lint
```

## Reproducing the paper

[REPRODUCING.md](REPRODUCING.md) maps each table and control to its build, training, and evaluation scripts, and lists the configuration. `scripts/train_method.py` defaults to the paper's main configuration.

The experiments ran on the Penn State ROAR cluster (A100 40 GB and A40 45 GB GPUs) between 2026-06-13 and 2026-06-27. The cluster's retention policy has since purged the built corpus, the RAG index, the checkpoints, and the per-prompt evaluation streams. Reproducing the paper therefore means rebuilding the corpus from the public sources and retraining.

## Related work

This work builds on *Scheduled Partial-Credit RL for Reliable Code Generation with Small Language Models* (LCTES 2026 WIP, [doi:10.1145/3814943.3816167](https://doi.org/10.1145/3814943.3816167)). Its code is at [SecureCodeRL](https://github.com/SuryanshSS1011/SecureCodeRL), tag `v0.1-lctes-final`. The two repositories share no git history.

## Citation

See [CITATION.cff](CITATION.cff). The code is archived on Zenodo: [10.5281/zenodo.23008092](https://doi.org/10.5281/zenodo.23008092) for all versions, [10.5281/zenodo.23008093](https://doi.org/10.5281/zenodo.23008093) for v1.0.0.

## License

MIT, see [LICENSE](LICENSE).
