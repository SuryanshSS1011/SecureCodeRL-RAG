# Reproducing CARGO

All paths below are relative to the repository root. The SLURM launchers (`scripts/queue_*.sh`, `scripts/*_watcher.sh`) were written for the ROAR cluster. Before running them, edit the `REPO`, `SCRATCH`, and `--account` values at the top of each file.

## 1. Build the v0.1.7 corpus

v0.1.7 has 3,150 train, 320 validation, and 1,582 evaluation prompts, plus 12,368 exemplar pairs. It is built in stages:

| Step | Script | Output |
|---|---|---|
| Export CVEfixes v1.0.8 SQLite to JSONL | `export_cvefixes_to_jsonl.py` | `raw/cvefixes_jsonl/` |
| Convert SecCodePLT parquet | `convert_seccodeplt_parquet_to_jsonl.py` | `raw/seccodeplt_jsonl/` |
| Per-CWE supply and quotas | `per_cwe_yield_v0_1_5.py`, `compute_v0_1_5_quotas.py` | quota JSON |
| Base build (run with `--build-version v0.1.6`) | `build_v0_1_5.py` | `build/v0.1.6/` |
| Author CWE-306 / CWE-862 design pairs | `author_design_pair_corpus.py` | `design_pair_patterns.jsonl` |
| Design-pair split and SFT pairs | `build_v0_1_5_1_corpus.py` | `build/v0.1.5.1/` (copy `design_pair_patterns.jsonl` into it) |
| Juliet 1.3 evaluation items | `extract_juliet_eval_items.py` | Juliet eval JSONL |
| Rebalance to v0.1.7 | `build_v0_1_7_rebalance.py` | `build/v0.1.7/` |
| Prompt splice for CyberSecEval / CWEval | `backfill_cyberseceval_cweval_splice.py` | patches `build/v0.1.7/` in place |
| Near-duplicate audit (Section V-A) | `measure_near_duplicate_distribution.py` | similarity report |
| Hybrid retrieval index | `build_rag_index.py` | `build/v0.1.7/rag_index/` |

`data/nvdlib_cwe_medians.json`, the per-CWE CVSS fallback, was produced by `build_nvdlib_medians.py`.

## 2. Train

Every RL cell runs `scripts/train_method.py`. The launcher for each paper artifact:

| Paper artifact | Launcher | Cells |
|---|---|---|
| Table IV, GRPO / PPO / RLOO / RAFT × {SAST only, + R_RAG}; Section II sweep | `queue_v0_1_7_factorial.sh` | `arm_a_{grpo,ppo,rloo,raft}[_rag]` |
| Table III, − CWE reweight | `queue_v0_1_7_controls.sh` | `arm_a_grpo_uniform_rag` |
| Table III, − σ-floor; σ_min sweep {0, 0.01, 0.1} and 0.1 → 0.01 anneal | `queue_post_headline_bundle.sh` | `sigma_floor_0`, `sigma_floor_0p01`, `sigma_floor_0p1`, `sigma_floor_anneal` |
| Seeds 1337 and 2024 | `queue_post_headline_bundle.sh` | `seed_1337`, `seed_2024` |
| Retrieval controls: random, adversarial, binary, binary + random | `queue_post_headline_bundle.sh` | `tierC1_random_rag`, `rag_adversarial`, `rag_binary`, `rag_binary_random` |
| Table V, Qwen2.5-Coder-3B and StarCoder2-3B | `queue_post_headline_bundle.sh` | `tierE1_qwen3b`, `tierE2_starcoder3b` |
| Table III, + SFT warm start | `train_sft.py --total-steps 20000`, then `queue_v0_1_7_sft_warm_start.sh` | `tierC2_sft_warm` |
| Table II, SFT-only and the LoRA-rank / full-FT ladder | `extract_sft_pairs.py`, `train_sft.py` via `sft_consolidated_watcher.sh` | `v3_sft_r*`, `v3_sft_fullft` |

## 3. Evaluate

| Paper artifact | Script |
|---|---|
| Table II, untrained and security-trained baselines | `run_baseline_sweep.py` via `queue_v0_1_7_baselines.sh` |
| Trained cells | `headline_eval.py` via `rl_headline_eval_watcher.sh` |
| Inference-time retrieval control (Section VI-C) | `defense_base_eval.py` |
| Compile-only rescoring of stored streams (Compile@1 definition, Section V-B) | `rescore_aggregates.py` |
| Section II zero-variance and nonzero-loss rates | `monitor_factorial_cells.py` |
| Seed mean ± std | `multiseed_summary.py` |
| Paired McNemar tests | `per_cwe_mcnemar.py` (run by `run_post_eval_analyses.sh`) |
| Table VI, offline retrieval-reward comparison (Section VI-F) | `offline_retrieval_reward.py` on Juliet C pairs, the authored design pairs, and PrimeVul paired fixes, with bge-base, UniXcoder, and CodeBERT |

Evaluation is greedy with at most 512 new tokens. Compile@1 uses `--compile-mode syntax_only`, which parses Python and runs `-fsyntax-only` for C/C++. Functional-Secure@1 is `func_sec_at_1__has_tests` in `eval/metrics.py`. Its denominator is fixed at the test-equipped subset for every system.

Table I is reproduced exactly by `tests/test_reward.py::test_reward_by_rollout_state_matches_table_one`. No script in this repository produces the held-out CWE figures or the compute totals.

## 4. Configuration

`scripts/train_method.py` defaults to the paper's main configuration (Sections IV and V-D), and the launchers override only the knob each cell varies.

| Setting | Value | Code |
|---|---|---|
| Reward (Eq. 4) | r = α_mix R_sec + (1 − α_mix) R_rel + λ_rag R_RAG − β 1[stub] | `reward/calculator.py` |
| α_mix, λ_rag, β | 0.3, 0.1, 1.5 | `--alpha`, `--lambda-rag`, `--stub-penalty` |
| R_sec | 0 if the rollout does not parse, else max(0, 1 − Σ CVSS/10 × confidence) | `reward/calculator.py` |
| R_RAG (Eq. 2) | cos(φ(y), φ(e⁺)), zeroed when the cosine exceeds τ_copy = 0.95 | `rag/r_rag.py` |
| Binary control | 1[cos(φ(y), φ(e⁺)) > cos(φ(y), φ(e⁻))] | `--rag-binary` |
| CWE reweighting (Eq. 3) | w(x) = (\|C\| / n_c)^α_cwe / Z, α_cwe = 0.5, mean 1 per batch | `rl/reweight.py`, `--alpha-cwe` |
| σ-floor | A_i = (r_i − r̄) / max(σ_r, σ_min), σ_min = 0.05 | `rl/grpo.py`, `--sigma-floor` |
| KL coefficient | 0.05 against the frozen base policy | `--kl-beta` |
| Optimizer | AdamW, peak LR 1e-5, cosine decay | `--learning-rate`, `--lr-schedule` |
| Batch | 4 prompts × G = 16 rollouts (G = 8 for PPO, SFT warm start, and the 3B policies) | `--batch-prompts`, `--group-size` |
| Policy | Qwen2.5-Coder-1.5B-Instruct, LoRA r = 16 on the attention projections | `--model-id`, `--lora-r` |
| Steps | 1000 | `--total-steps` |
| Analyzer timeout | 60 s per tool invocation, 120 s for CodeQL on C/C++ | `sast/runner.py` |
| Evaluation | greedy, at most 512 new tokens, Wilson 95% intervals | `eval/harness.py`, `eval/metrics.py` |
