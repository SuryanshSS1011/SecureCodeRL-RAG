"""SFT-only training on the v0.1.7 corpus's gold-pair subset (#262).

The SFT-only baseline defends against the reviewer attack "your headline
comes from the corpus, not the method." If supervised fine-tuning on
the same (prompt, secure_completion) pairs matches our RL headline,
then GRPO + R_RAG + reweighting adds no real value. If SFT-only falls
clearly short, the RL recipe is the load-bearing piece.

Implementation choices:
    - Same base model as RL: Qwen2.5-Coder-1.5B-Instruct
    - Same LoRA config as RL: r=16, alpha=32, dropout=0.05
    - Same LR schedule: cosine with 10% warmup, min_lr_rate=0.1
    - Cross-entropy loss masked to the completion tokens (NOT the prompt).
      Standard SFT recipe; computing loss over the prompt would dilute
      the gradient signal and reward the model for reproducing the
      prompt instead of the fix.
    - Single epoch over the SFT pair set. v0.1.7 SFT pairs are ~2,250
      so single-epoch at batch=4 = ~560 steps.
    - Saves to <output>/adapter/ matching the RL training output layout
      so the downstream Tier C2 SFT-warm-start cell can --resume-from-sft
      this directory.

Usage:
    PYTHONPATH=src .venv/bin/python scripts/train_sft.py \\
        --sft-pairs /scratch/.../v0.1.7/train_sft_pairs.jsonl \\
        --output /scratch/.../v0_1_7_sft_only \\
        --total-steps 560 \\
        --batch-size 4 \\
        --learning-rate 1e-5 \\
        --lora-r 16 --lora-alpha 32 \\
        --seed 42
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)


def _load_sft_pairs(path: Path) -> list[dict]:
    out = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _build_lr_lambda(total_steps: int, warmup_ratio: float, min_lr_rate: float):
    """Same cosine-with-warmup shape as torch_policy._build_cosine_with_warmup."""
    warmup_steps = max(1, int(warmup_ratio * total_steps))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / float(warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(1.0, progress)
        cos = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_rate + (1.0 - min_lr_rate) * cos

    return lr_lambda


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--sft-pairs", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model-id", type=str, default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--torch-dtype", type=str, default="bfloat16")
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--learning-rate", type=float, default=1e-5,
                   help="SFT uses higher LR than RL (1e-5 vs 1e-6 RL) since "
                        "the loss is cross-entropy not policy gradient.")
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--min-lr-rate", type=float, default=0.1)
    p.add_argument("--total-steps", type=int, default=560)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum-steps", type=int, default=1)
    p.add_argument("--max-seq-len", type=int, default=1024,
                   help="Truncate (prompt + completion) to this many tokens. "
                        "Anything longer is dropped silently.")
    p.add_argument("--save-every", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-fraction", type=float, default=0.1,
                   help="Fraction of pairs held out for val-loss-driven "
                        "checkpoint-best selection. 0.0 disables (falls back "
                        "to rolling-train-loss best).")
    p.add_argument("--val-every", type=int, default=100,
                   help="Run val loss eval every N steps (cheap; ~5s per pass).")
    p.add_argument("--resume-from", type=Path, default=None,
                   help="STRICT resume from a checkpoint-N directory written "
                        "by a recent run of this script. Requires all of "
                        "adapter/, optimizer.pt, and training_state.pt; "
                        "raises FileNotFoundError if any is missing (use "
                        "--warm-start-adapter instead for older checkpoints "
                        "that lack the state files). Resumes step counter, "
                        "AdamW moments, RNG; scheduler is rebuilt against "
                        "--total-steps and fast-forwarded. Logs appended.")
    p.add_argument("--warm-start-adapter", type=Path, default=None,
                   help="Load LoRA adapter weights from a checkpoint dir; "
                        "step counter starts at 0; AdamW + RNG cold. Used "
                        "when no optimizer.pt / training_state.pt exists "
                        "(e.g. checkpoints produced by an old commit). "
                        "This is a DIFFERENT EXPERIMENT than --resume-from "
                        "and must be reported as a warm-start in any paper "
                        "claim.")
    args = p.parse_args()

    # Validate flags BEFORE doing any heavy work (file loading, model
    # download). Mutual-exclusion check belongs here so the user gets a
    # clean error immediately, not after the SFT pair file is parsed.
    if args.resume_from is not None and args.warm_start_adapter is not None:
        p.error(
            "--resume-from and --warm-start-adapter are mutually exclusive"
        )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stderr,
    )

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "config.json").write_text(
        json.dumps(
            {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            indent=2,
        )
    )

    all_pairs = _load_sft_pairs(args.sft_pairs)
    logger.info("loaded %d SFT pairs from %s", len(all_pairs), args.sft_pairs)
    if not all_pairs:
        logger.error("empty SFT pair set; nothing to train")
        return 1

    # Train/val split. Use a deterministic shuffle keyed off args.seed so
    # the same split is reproducible across reruns and so the val set
    # doesn't leak into the train batches.
    import random as _random
    split_rng = _random.Random(args.seed)
    shuffled = list(all_pairs)
    split_rng.shuffle(shuffled)
    if args.val_fraction > 0:
        n_val = max(1, int(len(shuffled) * args.val_fraction))
        val_pairs = shuffled[:n_val]
        remainder = shuffled[n_val:]
        # Train-probe shard: a held-out slice of TRAIN data used only for
        # loss measurement (never gradient-trained on). Same size as the
        # val set so the per-pair noise floor matches val_loss and the
        # train/val gap reading is honest. Comparing train_probe_loss to
        # val_loss is the canonical SFT overfit signal: if train_probe
        # drops while val stalls/rises, the model is memorizing
        # train-distribution patterns rather than learning generalizable
        # ones. Critical at high LoRA rank (r=64+) where 4,648 pairs +
        # tens of millions of params makes overfit a real risk.
        n_probe = min(n_val, len(remainder) // 10)  # cap at 10% of remainder
        train_probe_pairs = remainder[:n_probe]
        pairs = remainder[n_probe:]
        logger.info(
            "train/val/train_probe split: %d train / %d val / %d train_probe "
            "(val_fraction=%.3f)",
            len(pairs), len(val_pairs), len(train_probe_pairs), args.val_fraction,
        )
    else:
        pairs = shuffled
        val_pairs = []
        train_probe_pairs = []
        logger.info("no val split; using rolling-train-loss for checkpoint-best")

    # Imports deferred so the script can be smoke-tested without torch.
    import numpy as np
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    logger.info("loading tokenizer + model: %s", args.model_id)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.torch_dtype]
    model = AutoModelForCausalLM.from_pretrained(args.model_id, dtype=dtype)
    model = model.to(args.device)

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate)
    lr_lambda = _build_lr_lambda(args.total_steps, args.warmup_ratio, args.min_lr_rate)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    # Resume: load adapter weights + optimizer + RNG + step counter from
    # a prior checkpoint. The scheduler is REBUILT against the new
    # args.total_steps above (not the previous run's), then fast-forwarded
    # to start_step so cosine decay runs cleanly to the new end. Logs
    # are appended to (not truncated) to preserve trajectory continuity.
    #
    # Two distinct modes via --resume-from / --warm-start-adapter:
    #   --resume-from <ckpt>      : strict; requires optimizer.pt +
    #                               training_state.pt + adapter/; raises
    #                               if any is missing. Used to continue a
    #                               run with proper data-ordering and
    #                               optimizer-state continuity.
    #   --warm-start-adapter <ck> : loose; loads adapter weights only,
    #                               starts step counter at 0, cold AdamW.
    #                               Explicitly distinct from --resume so
    #                               we never silently degrade a resume
    #                               into a warm-start (the 2026-06-20
    #                               incident where this script logged
    #                               "resume" but actually did a cold
    #                               warm-start and the operator reported
    #                               it as a resume).
    # Mutual-exclusion of --resume-from / --warm-start-adapter is
    # validated above at parse-args time.
    start_step = 0
    if args.resume_from is not None:
        resume_adapter = args.resume_from / "adapter"
        opt_path = args.resume_from / "optimizer.pt"
        state_path = args.resume_from / "training_state.pt"
        missing = [
            str(p) for p in (resume_adapter / "adapter_config.json",
                             opt_path, state_path) if not p.exists()
        ]
        if missing:
            raise FileNotFoundError(
                "--resume-from refuses to silently degrade to a cold "
                "warm-start. The following required files are missing: "
                + ", ".join(missing)
                + ". Either point --resume-from at a checkpoint produced "
                "by a recent train_sft.py (commit 2885052 or later), or "
                "use --warm-start-adapter to do an explicit cold-AdamW "
                "warm-start from the adapter weights only."
            )
        from peft import PeftModel  # noqa: F401  (import side effect)
        model.load_adapter(str(resume_adapter), adapter_name="default")
        optimizer.load_state_dict(torch.load(opt_path, map_location=args.device))
        st = torch.load(state_path, map_location="cpu")
        start_step = int(st["step"])
        rng = np.random.default_rng()
        rng.bit_generator.state = st["rng_state_np"]
        torch.set_rng_state(st["rng_state_torch"])
        if st.get("rng_state_cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(st["rng_state_cuda"])
        # Fast-forward the scheduler so lr matches the resume step.
        for _ in range(start_step):
            scheduler.step()
        logger.info(
            "RESUMED from step %d (adapter + optimizer + RNG); "
            "scheduler fast-forwarded (lr=%.2e); training to step %d.",
            start_step, optimizer.param_groups[0]["lr"], args.total_steps,
        )
    elif args.warm_start_adapter is not None:
        warm_adapter = args.warm_start_adapter / "adapter"
        if not (warm_adapter / "adapter_config.json").exists():
            raise FileNotFoundError(
                f"--warm-start-adapter {args.warm_start_adapter}: "
                "missing adapter/adapter_config.json"
            )
        model.load_adapter(str(warm_adapter), adapter_name="default")
        logger.info(
            "WARM-STARTED from %s (adapter weights only; AdamW cold; "
            "RNG cold; step counter starts at 0; training to step %d).",
            warm_adapter, args.total_steps,
        )

    log_mode = "a" if start_step > 0 else "w"
    train_log_fh = (args.output / "train_log.jsonl").open(log_mode)
    t0 = time.monotonic()

    def _build_batch(rows: list[dict]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Tokenize (prompt + completion), build labels masked to completion only.

        Returns (input_ids, attention_mask, labels). labels[i, j] = -100
        for prompt tokens and pad tokens; the completion tokens are
        kept so cross-entropy is computed only on them.
        """
        texts = []
        prompt_lens = []
        for row in rows:
            prompt_text = row["prompt_text"]
            secure = row["secure_completion"]
            # Standard SFT formatting: prompt then completion, separated
            # by a newline. The tokenizer's chat template would also work
            # but we match the RL trainer's plain-prompt format.
            full = prompt_text.rstrip() + "\n" + secure
            texts.append(full)
            # Length of the prompt portion (we need this to mask labels).
            prompt_ids = tokenizer(
                prompt_text.rstrip() + "\n",
                add_special_tokens=False,
            )["input_ids"]
            prompt_lens.append(len(prompt_ids))

        encoded = tokenizer(
            texts,
            max_length=args.max_seq_len,
            truncation=True,
            padding="longest",
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        labels = input_ids.clone()
        for i, pl in enumerate(prompt_lens):
            # Mask the prompt and any pad tokens.
            labels[i, :pl] = -100
        labels[attention_mask == 0] = -100
        return input_ids, attention_mask, labels

    # checkpoint-best tracking. Primary signal is held-out val loss
    # (--val-fraction > 0). Fallback is rolling-window train loss when
    # no val set is held out.
    train_log_buf: list[dict] = []
    best_val_loss: float = float("inf")
    best_step: int = 0
    val_log_fh = (args.output / "val_log.jsonl").open(log_mode) if val_pairs else None

    @torch.no_grad()
    def _compute_loss_on(probe_pairs: list[dict]) -> float:
        """Mean cross-entropy on a fixed held-out shard. Same pair order
        and batch composition every call → trajectories are comparable
        across steps. Used for both val_loss (generalization signal) and
        train_probe_loss (overfit-detection signal). Compute is symmetric
        so the train_probe / val_loss gap reading is honest."""
        if not probe_pairs:
            return float("inf")
        model.eval()
        total_loss = 0.0
        total_batches = 0
        for batch_start in range(0, len(probe_pairs), args.batch_size):
            batch = probe_pairs[batch_start : batch_start + args.batch_size]
            if not batch:
                continue
            iids, mask, lbls = _build_batch(batch)
            iids = iids.to(args.device)
            mask = mask.to(args.device)
            lbls = lbls.to(args.device)
            out = model(input_ids=iids, attention_mask=mask, labels=lbls)
            total_loss += float(out.loss.detach().item())
            total_batches += 1
        model.train()
        return total_loss / max(1, total_batches)

    for step in range(start_step, args.total_steps):
        idx = rng.integers(0, len(pairs), size=args.batch_size)
        batch = [pairs[i] for i in idx]
        input_ids, attention_mask, labels = _build_batch(batch)
        input_ids = input_ids.to(args.device)
        attention_mask = attention_mask.to(args.device)
        labels = labels.to(args.device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = outputs.loss
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        rec = {
            "step": step,
            "loss": float(loss.detach().item()),
            "grad_norm": float(grad_norm),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "wall_s_step": time.monotonic() - t0,
        }
        train_log_fh.write(json.dumps(rec) + "\n")
        train_log_fh.flush()
        train_log_buf.append(rec)

        # Periodic val + train_probe eval (cheap; ~5-10s combined).
        # Logging both lets readers spot overfit by train_probe << val
        # divergence — the canonical signal that LoRA capacity has
        # outgrown the data and the model is memorizing.
        if val_pairs and (step + 1) % args.val_every == 0:
            v_loss = _compute_loss_on(val_pairs)
            tp_loss = _compute_loss_on(train_probe_pairs) if train_probe_pairs else float("nan")
            gap = (v_loss - tp_loss) if train_probe_pairs else float("nan")
            val_rec = {
                "step": step + 1,
                "val_loss": v_loss,
                "train_probe_loss": tp_loss,
                "overfit_gap": gap,  # val - train_probe: positive = overfit
            }
            val_log_fh.write(json.dumps(val_rec) + "\n")
            val_log_fh.flush()
            if train_probe_pairs:
                logger.info(
                    "step %d val_loss=%.4f train_probe_loss=%.4f gap=%+.4f",
                    step + 1, v_loss, tp_loss, gap,
                )
            else:
                logger.info("step %d val_loss=%.4f", step + 1, v_loss)

        if (step + 1) % args.save_every == 0 or (step + 1) == args.total_steps:
            ck = args.output / f"checkpoint-{step + 1}"
            ck.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(ck / "adapter"))
            # Save optimizer + training state so a future --resume-from
            # can continue without restarting AdamW moments cold and
            # without re-shuffling the prompt ordering.
            torch.save(optimizer.state_dict(), ck / "optimizer.pt")
            torch.save({
                "step": step + 1,
                "rng_state_np": rng.bit_generator.state,
                "rng_state_torch": torch.get_rng_state(),
                "rng_state_cuda": (
                    torch.cuda.get_rng_state_all()
                    if torch.cuda.is_available() else None
                ),
            }, ck / "training_state.pt")
            logger.info("saved checkpoint at step %d to %s", step + 1, ck)

            # Checkpoint-best selection. Primary: held-out val loss
            # (computed just above when --val-fraction > 0). Fallback:
            # rolling mean of last save_every train losses.
            if val_pairs:
                # Use the val_loss just computed (val_every is aligned to
                # save_every here; if not, recompute).
                if (step + 1) % args.val_every != 0:
                    v_loss = _compute_loss_on(val_pairs)
                else:
                    v_loss = val_rec["val_loss"]  # type: ignore[possibly-undefined]
                if v_loss < best_val_loss:
                    best_val_loss = v_loss
                    best_step = step + 1
                    best_link = args.output / "checkpoint-best"
                    if best_link.exists() or best_link.is_symlink():
                        best_link.unlink()
                    best_link.symlink_to(ck.name)
                    logger.info(
                        "checkpoint-best -> checkpoint-%d (val_loss=%.4f)",
                        step + 1, v_loss,
                    )
            else:
                recent = [r["loss"] for r in train_log_buf[-args.save_every:]]
                window_loss = sum(recent) / max(1, len(recent))
                if window_loss < best_val_loss:
                    best_val_loss = window_loss
                    best_step = step + 1
                    best_link = args.output / "checkpoint-best"
                    if best_link.exists() or best_link.is_symlink():
                        best_link.unlink()
                    best_link.symlink_to(ck.name)
                    logger.info(
                        "checkpoint-best -> checkpoint-%d (window_train_loss=%.4f)",
                        step + 1, window_loss,
                    )

    # Final adapter location matches the RL training output: <output>/adapter
    model.save_pretrained(str(args.output / "adapter"))
    train_log_fh.close()
    if val_log_fh is not None:
        val_log_fh.close()
    selection_signal = "val_loss" if val_pairs else "window_train_loss"
    logger.info(
        "SFT-only training complete; final adapter at %s/adapter; "
        "checkpoint-best -> checkpoint-%d (%s=%.4f)",
        args.output, best_step, selection_signal, best_val_loss,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
