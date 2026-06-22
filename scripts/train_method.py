#!/usr/bin/env python3
"""Train one RL cell of the CARGO paper.

This entrypoint wires:
  - TorchPolicy (Qwen2.5-Coder-1.5B-Instruct + LoRA r=16 on the attention
    projections)
  - RewardPipeline (RealOracle + CodeQL / Semgrep / Bandit / Cppcheck),
    composed per Eq. 4, with R_RAG (Eq. 2) when --reward-rag-on
  - GRPO / PPO / RLOO / RAFT via --algorithm, with the sigma floor
    (Section IV-C) and CWE-aware per-prompt weights (Eq. 3)
  - Trainer.run() over train_prompts.jsonl

Defaults are the paper's main configuration (Section V-D): 1000 steps,
G = 16, 4 prompts per step, AdamW with cosine decay and peak LR 1e-5,
alpha_mix = 0.3, lambda_rag = 0.1, beta = 1.5, sigma_min = 0.05,
alpha_cwe = 0.5, beta_KL = 0.05.

Output:
  - <output-dir>/train_log.jsonl    one row per training step
  - <output-dir>/checkpoint/        final LoRA adapter

CARGO (GRPO + R_RAG):
    python scripts/train_method.py \\
        --train-jsonl build/v0.1.7/train_prompts.jsonl \\
        --output runs/cargo_grpo \\
        --reward-rag-on --rag-index-dir build/v0.1.7/rag_index
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Optional

from secure_code_rl_ictai.data_prep import normalize_cwe, normalize_language
from secure_code_rl_ictai.data_prep.schema import Prompt
from secure_code_rl_ictai.reward import (
    MockOracle,
    RealOracle,
    ReliabilitySignals,
    RewardCalculator,
    RewardConfig,
    RewardPipeline,
    TestCase,
    TestSpec,
)
from secure_code_rl_ictai.rl.grpo import GrpoConfig
from secure_code_rl_ictai.rl.registry import get_algorithm
from secure_code_rl_ictai.rl.reweight import (
    DEFAULT_ALPHA_CWE,
    ReweightConfig,
    Reweighter,
)
from secure_code_rl_ictai.rl.schedule import DEFAULT_PHASES, PhaseSchedule, PhaseSpec
from secure_code_rl_ictai.rl.torch_policy import TorchPolicy, TorchPolicyConfig
from secure_code_rl_ictai.rl.trainer import Trainer, TrainerConfig
from secure_code_rl_ictai.sast.adapters.bandit import BanditAdapter
from secure_code_rl_ictai.sast.adapters.codeql import CodeQLAdapter
from secure_code_rl_ictai.sast.adapters.cppcheck import CppcheckAdapter
from secure_code_rl_ictai.sast.adapters.semgrep import SemgrepAdapter
from secure_code_rl_ictai.sast.models import ToolName
from secure_code_rl_ictai.sast.runner import MockAdapter, SastRunner
from secure_code_rl_ictai.sast.severity import SeveritySource


def _resolve(name: str) -> str:
    via_path = shutil.which(name)
    if via_path:
        return via_path
    candidate = Path(sys.executable).parent / name
    if candidate.exists():
        return str(candidate)
    return name


def _build_reward_pipeline(
    *,
    sast_real: bool,
    oracle_real: bool,
    alpha: float,
    retriever=None,
    embedder=None,
    lambda_rag: float = 0.0,
    copy_guard_threshold: float = 0.95,
    stub_penalty: float = 0.0,
    rag_binary: bool = False,
) -> RewardPipeline:
    """Construct the RewardPipeline used during training.

    `sast_real=True`: Bandit + Semgrep + CodeQL + Cppcheck.
        Required for any cell that uses R_security (cells 3 and 4).
    `oracle_real=True`: RealOracle compiles + runs code in a subprocess.
        Required for R_reliability to be informative.
    """
    if sast_real:
        bandit_adapter = BanditAdapter(bandit_binary=_resolve("bandit"))
        semgrep_adapter = SemgrepAdapter(semgrep_binary=_resolve("semgrep"))

        codeql_bin = os.environ.get("CODEQL_BINARY") or _resolve("codeql")
        codeql_adapter = CodeQLAdapter(codeql_binary=codeql_bin)

        cppcheck_bin = os.environ.get("CPPCHECK_BINARY") or _resolve("cppcheck")
        cppcheck_adapter = CppcheckAdapter(cppcheck_binary=cppcheck_bin)
    else:
        bandit_adapter = MockAdapter(ToolName.BANDIT, {"runs": []})
        semgrep_adapter = MockAdapter(ToolName.SEMGREP, {"runs": []})
        codeql_adapter = MockAdapter(ToolName.CODEQL, {"runs": []})
        cppcheck_adapter = MockAdapter(ToolName.CPPCHECK, {"runs": []})

    adapters = {
        ToolName.CODEQL: codeql_adapter,
        ToolName.SEMGREP: semgrep_adapter,
        ToolName.BANDIT: bandit_adapter,
        ToolName.CPPCHECK: cppcheck_adapter,
    }
    # CHEAP_PERIODIC: fast tools (semgrep, bandit, cppcheck) every step;
    # deep tools (codeql) every 50th step. Codeql analyze takes
    # ~90s on cold start; running on every rollout would make training
    # 17 days/run. With deep_period=50 it's amortized to ~5min per step
    # average, ~12h per 1000-step run. Full codeql signal returned at
    # eval time (headline_eval.py uses SastTier.ALL).
    from secure_code_rl_ictai.sast.runner import SastTier
    runner = SastRunner(adapters, tier=SastTier.CHEAP_PLUS_PERIODIC, deep_period=50)
    sev_src = SeveritySource(Path("data/nvdlib_cwe_medians.json"))
    calc = RewardCalculator(
        RewardConfig(alpha=alpha, stub_penalty=stub_penalty), sev_src
    )

    if oracle_real:
        # Resolve C/C++ compilers per host. ROAR has gcc/g++ but not clang.
        c_cc = _resolve("clang") if shutil.which("clang") else _resolve("gcc")
        cpp_cc = _resolve("clang++") if shutil.which("clang++") else _resolve("g++")
        oracle = RealOracle(
            python_executable=_resolve("python3"),
            c_compiler=c_cc,
            cpp_compiler=cpp_cc,
        )
    else:
        oracle = MockOracle(
            ReliabilitySignals(compiles=True, tests_passed=1, tests_total=1)
        )

    return RewardPipeline(
        oracle=oracle,
        sast_runner=runner,
        severity_source=sev_src,
        calculator=calc,
        retriever=retriever,
        embedder=embedder,
        lambda_rag=lambda_rag,
        copy_guard_threshold=copy_guard_threshold,
        rag_binary=rag_binary,
    )


def _load_prompts(path: Path, n: int) -> list[Prompt]:
    """Load up to n prompts (n=0 means all)."""
    out: list[Prompt] = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            lang = normalize_language(rec["language"])
            ts = rec.get("test_spec", {})
            test_spec = TestSpec(
                language=lang,
                test_cases=[
                    TestCase(
                        input_stdin=str(tc.get("input_stdin", "")),
                        expected_stdout=str(tc.get("expected_stdout", "")),
                        timeout_s=float(tc.get("timeout_s", 5.0)),
                    )
                    for tc in ts.get("test_cases", [])
                ],
                extra_files=dict(ts.get("extra_files", {}) or {}),
                entry_module=ts.get("entry_module"),
                prefix_text=ts.get("prefix_text"),
                suffix_text=ts.get("suffix_text"),
            )
            out.append(
                Prompt(
                    id=rec["id"],
                    source=rec.get("source", "unknown"),
                    language=lang,
                    target_cwe=normalize_cwe(rec["target_cwe"]),
                    prompt_text=rec["prompt_text"],
                    test_spec=test_spec,
                    task_signature=rec.get("task_signature"),
                    metadata=rec.get("metadata", {}),
                )
            )
            if n > 0 and len(out) >= n:
                break
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    # Data
    p.add_argument("--train-jsonl", type=Path, required=True)
    p.add_argument("--max-prompts", type=int, default=0,
                   help="Cap on number of training prompts (0 = use all).")
    # Output
    p.add_argument("--output", type=Path, required=True,
                   help="Output dir for train_log.jsonl + checkpoint + config.")
    # Model
    p.add_argument("--model-id", type=str, default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--torch-dtype", type=str, default="bfloat16")
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--learning-rate", type=float, default=1e-5)
    # "cosine" adds linear warmup over --warmup-ratio * total-steps, then
    # cosine decay to --min-lr-rate * learning-rate.
    p.add_argument("--lr-schedule", type=str, default="cosine",
                   choices=("constant", "cosine"),
                   help="LR schedule. cosine adds 10%% warmup then decay.")
    p.add_argument("--warmup-ratio", type=float, default=0.1,
                   help="Fraction of total-steps spent in linear warmup. "
                        "Only used when --lr-schedule=cosine.")
    p.add_argument("--min-lr-rate", type=float, default=0.1,
                   help="Floor for cosine decay as a multiplier on "
                        "learning-rate. 0.1 matches Open-RS / Tulu 3.")
    # Trainer
    p.add_argument("--algorithm", type=str, default="grpo",
                   choices=("grpo", "ppo", "rloo", "raft"))
    # PPO value head (only used when --algorithm=ppo). 0.5 matches the
    # InstructGPT/RLHF default. Value-head lr can differ from policy lr;
    # None means "use the policy learning rate".
    p.add_argument("--value-loss-weight", type=float, default=0.5,
                   help="PPO value loss weight (only used when "
                        "--algorithm=ppo). Default 0.5.")
    p.add_argument("--value-head-lr", type=float, default=None,
                   help="PPO value head learning rate. None means "
                        "use --learning-rate.")
    p.add_argument("--total-steps", type=int, default=1000)
    p.add_argument("--batch-prompts", type=int, default=4)
    p.add_argument("--group-size", type=int, default=16)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--seed", type=int, default=42)
    # Checkpoint cadence (§P0.1) + resume (§P0.2). Default save_every=0 keeps
    # the legacy "save only at end" behavior for short runs; Arm A's queue
    # script sets save_every=250 so SLURM 11hr kills lose at most 250 steps.
    p.add_argument("--save-every", type=int, default=0,
                   help="Save a checkpoint every N steps. 0 = disabled "
                        "(legacy: save only at end). Arm A should use 250.")
    p.add_argument("--keep-last-k", type=int, default=3,
                   help="Rotate intermediate checkpoints: keep the last K "
                        "+ the final-step one. Bounds disk use on long runs.")
    p.add_argument("--resume-from", type=Path, default=None,
                   help="Resume training from a checkpoint-{N}/ directory "
                        "(restores LoRA adapter, optimizer state, step "
                        "counter, RNG state). If the path "
                        "is the output dir itself instead of a specific "
                        "checkpoint, the latest checkpoint-{N}/ inside it "
                        "is picked automatically. Requires trainer_state.json "
                        "(written by this script's checkpoint saves). For "
                        "loading SFT-trained adapter weights as RL init, "
                        "use --warm-start-adapter instead.")
    p.add_argument("--warm-start-adapter", type=Path, default=None,
                   help="Load LoRA adapter weights from a checkpoint dir "
                        "(typically produced by scripts/train_sft.py). RL "
                        "training then starts fresh from step 0 with cold "
                        "AdamW; the adapter just provides "
                        "the initialization. The path should point at a dir "
                        "that has an adapter/ subdir with adapter_config.json. "
                        "This is a DIFFERENT EXPERIMENT than --resume-from "
                        "(which assumes RL-side state files) and is the right "
                        "flag for Tier C2 (SFT warm-start vs cold-start RL).")
    # Training-time eval (§P0.3 / §P0.4). Cheap on ~30 val prompts; runs
    # at temperature 0 for determinism so eval curves are reproducible.
    p.add_argument("--eval-jsonl", type=Path, default=None,
                   help="Path to val_prompts.jsonl. When set together with "
                        "--eval-every > 0, the trainer runs eval at cadence "
                        "and writes eval_log.jsonl + symlinks checkpoint-best.")
    p.add_argument("--eval-every", type=int, default=0,
                   help="Run training-time eval every N steps. 0 = disabled.")
    p.add_argument("--eval-max-prompts", type=int, default=0,
                   help="Cap on number of val prompts per eval. 0 = use all. "
                        "Useful for smoke runs to keep eval wall under 60s.")
    # Early-stop tripwire (Open-RS / Dang & Ngo: SLM GRPO collapse becomes
    # irrecoverable past a few hundred degraded steps; we want a clean
    # abort that leaves checkpoint-best in place). 0 disables.
    p.add_argument("--early-stop-patience", type=int, default=0,
                   help="Abort training after this many consecutive evals "
                        "below (best - --early-stop-drop-pp). 0 = disabled. "
                        "Recommended 4 for 1000-step runs with eval-every-100.")
    p.add_argument("--early-stop-drop-pp", type=float, default=5.0,
                   help="Percent-point drop from best val metric that "
                        "counts as a degraded eval.")
    # Reward pipeline switches
    p.add_argument("--alpha", type=float, default=DEFAULT_PHASES[0].alpha,
                   help="alpha_mix: weight on R_sec in Eq. 4; R_rel gets 1 - alpha.")
    p.add_argument("--kl-beta", type=float, default=DEFAULT_PHASES[0].beta,
                   help="KL coefficient against the frozen base policy.")
    p.add_argument("--reward-rag-on", action="store_true",
                   help="Enable training-time RAG: each rollout's reward gains "
                        "lambda*R_RAG where R_RAG measures similarity to the "
                        "top-1 retrieved secure exemplar. Requires --rag-index-dir.")
    p.add_argument("--rag-index-dir", type=Path, default=None,
                   help="Path to built RAG index (pairs.jsonl, bm25.pickle, "
                        "faiss.index, embeddings.npy, manifest.json). "
                        "Built by scripts/build_rag_index.py.")
    p.add_argument("--lambda-rag", type=float, default=0.1,
                   help="lambda_rag: weight on R_RAG in Eq. 4.")
    p.add_argument("--rag-copy-guard-threshold", type=float, default=0.95,
                   help="Cosine-similarity cutoff above which R_RAG is masked "
                        "(prevents reward-hacking via copy-paste). Default 0.95.")
    p.add_argument("--stub-penalty", type=float, default=1.5,
                   help="beta in Eq. 4: subtracted from the reward of a "
                        "completion flagged as a stub.")
    # Retrieval controls (Section IV-A): adversarial = lowest-ranked
    # exemplar within the prompt's CWE; random = uniform over the index.
    p.add_argument("--rag-retriever-mode", type=str, default="best",
                   choices=("best", "adversarial", "random"),
                   help="best = top-ranked exemplar for the CWE; "
                        "adversarial = lowest-ranked within the CWE; "
                        "random = uniform over all exemplars.")
    # Binary control (Section IV-A): 1[cos(y, e+) > cos(y, e-)] in place
    # of the continuous cosine of Eq. 2.
    p.add_argument("--rag-binary", action="store_true",
                   help="Replace continuous R_RAG with binary indicator "
                        "(1 if cos(completion, e_pos) > cos(completion, e_neg)).")
    p.add_argument("--prepend-rag-on", action="store_true",
                   help="Enable training-time PROMPT-PREPEND RAG (distinct "
                        "from --reward-rag-on): before each rollout, prepend "
                        "the top-1 retrieved secure exemplar to prompt_text "
                        "as a reference block. The model SEES the exemplar. "
                        "Requires --rag-index-dir. May be combined with "
                        "--reward-rag-on (both contribute distinct signals).")
    p.add_argument("--no-sast", action="store_true",
                   help="Disable real SAST (use MockAdapters). R_security collapses "
                        "to the no-findings reward. Use for SFT-only smokes.")
    p.add_argument("--no-oracle", action="store_true",
                   help="Disable RealOracle (use MockOracle). R_reliability becomes "
                        "always-pass; useful for SAST-only debugging.")
    # CWE-aware per-prompt reweighting (Eq. 3).
    p.add_argument("--reweight-enabled", choices=("true", "false"), default="true",
                   help="false = uniform weights (the '- CWE reweight' ablation).")
    p.add_argument("--alpha-cwe", type=float, default=DEFAULT_ALPHA_CWE,
                   help="Exponent alpha_cwe in Eq. 3.")
    # sigma floor on the group-relative normalizer (Section IV-C).
    p.add_argument("--sigma-floor", type=float, default=0.05,
                   help="sigma_min: advantages divide by max(std, sigma_min). "
                        "0 disables the floor.")
    # Adaptive σ_floor schedule. When --sigma-floor-anneal-steps > 0, the
    # trainer linearly interpolates σ_floor from --sigma-floor-initial at
    # step 0 to --sigma-floor-final at the anneal step, held after. This
    # overrides the constant --sigma-floor while active.
    p.add_argument("--sigma-floor-initial", type=float, default=0.0,
                   help="Adaptive σ_floor starting value (step 0).")
    p.add_argument("--sigma-floor-final", type=float, default=0.0,
                   help="Adaptive σ_floor terminal value (after anneal-steps).")
    p.add_argument("--sigma-floor-anneal-steps", type=int, default=0,
                   help="Anneal σ_floor over this many steps. 0 = disabled "
                        "(use constant --sigma-floor instead).")
    # Entropy bonus with linear decay. Adds `entropy_coef * log_probs.mean()`
    # to the policy step's surrogate (standard PG-with-entropy form).
    p.add_argument("--entropy-coef-initial", type=float, default=0.0,
                   help="Initial entropy bonus coefficient.")
    p.add_argument("--entropy-coef-final", type=float, default=0.0,
                   help="Final entropy bonus coefficient (after decay-steps).")
    p.add_argument("--entropy-decay-steps", type=int, default=0,
                   help="Linear decay window for entropy coef. 0 = use "
                        "--entropy-coef-initial as a constant.")
    args = p.parse_args()

    # Validate mutually-exclusive resume modes BEFORE any heavy work
    # (model download, RAG index load). Mirrors the corresponding check
    # in scripts/train_sft.py.
    if args.resume_from is not None and args.warm_start_adapter is not None:
        p.error(
            "--resume-from and --warm-start-adapter are mutually exclusive"
        )

    args.output.mkdir(parents=True, exist_ok=True)

    # Persist the run config.
    (args.output / "config.json").write_text(
        json.dumps({k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                   indent=2)
    )

    # 1. Load prompts.
    print(f"[train] loading prompts from {args.train_jsonl} ...", file=sys.stderr, flush=True)
    prompts = _load_prompts(args.train_jsonl, args.max_prompts)
    print(f"[train] loaded {len(prompts)} prompts", file=sys.stderr, flush=True)

    # 2. Build the policy (lazy-loads on first generate/log_probs/step).
    policy_config = TorchPolicyConfig(
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        learning_rate=args.learning_rate,
        lr_schedule=args.lr_schedule,
        warmup_ratio=args.warmup_ratio,
        min_lr_rate=args.min_lr_rate,
        # cosine scheduler needs total_steps; constant ignores it.
        total_steps=args.total_steps if args.lr_schedule == "cosine" else None,
    )
    policy = TorchPolicy(
        model_id=args.model_id,
        device=args.device,
        torch_dtype=args.torch_dtype,
        config=policy_config,
    )

    # When PPO is requested, turn on the value head BEFORE the model
    # loads. Without this, ppo.py:47-48 silently falls back to REINFORCE
    # (advantages = rewards), which was the original PPO bug — the cell
    # ran the wrong algorithm under a PPO label. See FINDINGS_LOG
    # 2026-06-14.
    if args.algorithm == "ppo":
        policy.enable_value_head(
            value_loss_weight=args.value_loss_weight,
            value_head_lr=args.value_head_lr,
        )
        print(
            f"[train] PPO selected; value head enabled "
            f"(value_loss_weight={args.value_loss_weight}, "
            f"value_head_lr={args.value_head_lr or args.learning_rate})",
            file=sys.stderr, flush=True,
        )

    # 2b. Resume-from-checkpoint discovery (§P0.2). If --resume-from is the
    # output dir itself, find the most-advanced checkpoint-{N}/ inside it;
    # if it's a specific checkpoint-{N}/, use that. None if no checkpoint
    # exists. The selected dir is passed to the policy BEFORE _ensure_loaded,
    # so the LoRA adapter + optimizer are restored on first model use.
    start_step = 0
    if args.resume_from is not None:
        ckpt_dir = args.resume_from
        if not (ckpt_dir / "trainer_state.json").exists():
            ckpt_dir = Trainer.find_latest_checkpoint(args.resume_from)
        if ckpt_dir is None:
            # Refuse to silently degrade to a cold start. The user likely
            # meant --warm-start-adapter; pointing them at it is more
            # helpful than starting fresh and looking like a successful
            # resume in the log.
            raise FileNotFoundError(
                f"--resume-from={args.resume_from} has no checkpoint to "
                "resume from (no trainer_state.json found at the path or "
                "in any checkpoint-{N}/ subdir). If you meant to load only "
                "the LoRA adapter weights as init (e.g. from an SFT run), "
                "use --warm-start-adapter instead."
            )
        state = Trainer.load_trainer_state(ckpt_dir)
        start_step = int(state["step"])
        policy.mark_for_resume(ckpt_dir)
        print(f"[train] resuming from {ckpt_dir} at step {start_step}",
              file=sys.stderr, flush=True)
    elif args.warm_start_adapter is not None:
        # Warm-start: load LoRA adapter weights only. No optimizer, no
        # scheduler, no trainer_state. Step counter starts at 0. The
        # adapter dir must have adapter/adapter_config.json. The policy's
        # _ensure_loaded() path will tolerate the missing optimizer.pt /
        # scheduler.pt files (they are checked via if path.exists()).
        warm_dir = args.warm_start_adapter
        if not (warm_dir / "adapter" / "adapter_config.json").exists():
            raise FileNotFoundError(
                f"--warm-start-adapter={warm_dir}: missing "
                "adapter/adapter_config.json"
            )
        policy.mark_for_resume(warm_dir)
        print(
            f"[train] WARM-STARTING from {warm_dir} "
            f"(adapter weights only; AdamW + RNG cold; "
            f"step counter starts at 0; training to step {args.total_steps}).",
            file=sys.stderr, flush=True,
        )

    # 3. Build the reward pipeline.
    initial_alpha = args.alpha

    # P1.3 RAG cells. Two orthogonal switches:
    #   --reward-rag-on   : R_total gains lambda*R_RAG (model never sees exemplar)
    #   --prepend-rag-on  : prompt is wrapped with retrieved exemplar (model sees it)
    # Either or both load the retriever/embedder from --rag-index-dir.
    retriever = None
    embedder = None
    need_rag = args.reward_rag_on or args.prepend_rag_on
    if need_rag:
        if args.rag_index_dir is None:
            raise ValueError(
                "--reward-rag-on / --prepend-rag-on requires --rag-index-dir"
            )
        from secure_code_rl_ictai.rag import load_embedder, load_retriever
        print(
            f"[train] loading RAG index from {args.rag_index_dir} ...",
            file=sys.stderr, flush=True,
        )
        retriever = load_retriever(
            args.rag_index_dir,
            mode=args.rag_retriever_mode,
            rng_seed=args.seed,
        )
        embedder = load_embedder(args.rag_index_dir / "manifest.json")
        modes = []
        if args.reward_rag_on:
            modes.append(f"reward(lambda={args.lambda_rag})")
        if args.prepend_rag_on:
            modes.append("prepend")
        print(
            f"[train] RAG retriever loaded ({len(retriever.pairs)} pairs); "
            f"modes={modes}, copy_guard={args.rag_copy_guard_threshold}",
            file=sys.stderr, flush=True,
        )

    reward_pipeline = _build_reward_pipeline(
        sast_real=not args.no_sast,
        oracle_real=not args.no_oracle,
        alpha=initial_alpha,
        retriever=retriever,
        embedder=embedder,
        lambda_rag=args.lambda_rag if args.reward_rag_on else 0.0,
        copy_guard_threshold=args.rag_copy_guard_threshold,
        stub_penalty=args.stub_penalty,
        rag_binary=args.rag_binary,
    )

    # 4. Schedule.
    schedule = PhaseSchedule(
        (PhaseSpec(number=1, start_step=0, alpha=args.alpha, beta=args.kl_beta),)
    )

    # 5. Algorithm.
    grpo_config = GrpoConfig(sigma_floor=args.sigma_floor, kl_beta=args.kl_beta)
    algorithm = get_algorithm(args.algorithm, grpo_config)

    # 5b. CWE-aware per-prompt weights (Eq. 3) from training-pool counts.
    reweighter: Optional[Reweighter] = None
    if args.reweight_enabled == "true":
        reweighter = Reweighter.from_cwes(
            ReweightConfig(alpha_cwe=args.alpha_cwe),
            (p.target_cwe for p in prompts),
        )
        print(f"[train] CWE reweighting over {len(reweighter.trained_cwes)} CWEs "
              f"(alpha_cwe={args.alpha_cwe})", file=sys.stderr, flush=True)
    else:
        print("[train] reweighting disabled (uniform weights)",
              file=sys.stderr, flush=True)

    # 5c. Val prompts (§P0.3 training-time eval). Loaded only when both
    # --eval-jsonl and --eval-every are set; otherwise we skip the I/O so
    # smoke runs don't pay for the load.
    val_prompts = None
    if args.eval_jsonl is not None and args.eval_every > 0:
        val_prompts = _load_prompts(args.eval_jsonl, args.eval_max_prompts)
        print(f"[train] loaded {len(val_prompts)} val prompts for eval cadence",
              file=sys.stderr, flush=True)

    # 6. Trainer.
    trainer_config = TrainerConfig(
        total_steps=args.total_steps,
        batch_prompts=args.batch_prompts,
        group_size=args.group_size,
        seed=args.seed,
        log_path=args.output / "train_log.jsonl",
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        save_every=args.save_every,
        keep_last_k=args.keep_last_k,
        output_dir=args.output,
        eval_every=args.eval_every,
        eval_prompts=val_prompts,
        eval_max_prompts=args.eval_max_prompts,
        prepend_rag_on=args.prepend_rag_on,
        early_stop_patience=args.early_stop_patience,
        early_stop_drop_pp=args.early_stop_drop_pp,
        sigma_floor_initial=args.sigma_floor_initial,
        sigma_floor_final=args.sigma_floor_final,
        sigma_floor_anneal_steps=args.sigma_floor_anneal_steps,
        entropy_coef_initial=args.entropy_coef_initial,
        entropy_coef_final=args.entropy_coef_final,
        entropy_decay_steps=args.entropy_decay_steps,
    )
    trainer = Trainer(
        policy=policy,
        reward_pipeline=reward_pipeline,
        schedule=schedule,
        algorithm=algorithm,
        config=trainer_config,
        reweighter=reweighter,
    )

    # 7. Run. start_step > 0 when resuming; the trainer iterates
    # range(start_step, total_steps) and fast-forwards the RNG.
    t0 = time.perf_counter()
    remaining = args.total_steps - start_step
    print(f"[train] starting {remaining} steps from step {start_step} "
          f"to {args.total_steps} "
          f"({args.batch_prompts} prompts x {args.group_size} rollouts)",
          file=sys.stderr, flush=True)
    records = trainer.run(prompts, start_step=start_step)
    wall = time.perf_counter() - t0
    print(f"[train] done: {len(records)} steps in {wall:.1f}s "
          f"({wall / max(1, len(records)):.2f}s/step)", file=sys.stderr, flush=True)

    # 8. Save final LoRA adapter.
    ckpt = args.output / "checkpoint"
    try:
        if policy._model is not None and hasattr(policy._model, "save_pretrained"):
            policy._model.save_pretrained(str(ckpt))
            policy._tokenizer.save_pretrained(str(ckpt))
            print(f"[train] saved adapter to {ckpt}", file=sys.stderr, flush=True)
    except Exception as exc:
        print(f"[train] failed to save adapter: {exc}", file=sys.stderr, flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
