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
    args = p.parse_args()

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

    pairs = _load_sft_pairs(args.sft_pairs)
    logger.info("loaded %d SFT pairs from %s", len(pairs), args.sft_pairs)
    if not pairs:
        logger.error("empty SFT pair set; nothing to train")
        return 1

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

    train_log_fh = (args.output / "train_log.jsonl").open("w")
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

    for step in range(args.total_steps):
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

        if (step + 1) % args.save_every == 0 or (step + 1) == args.total_steps:
            ck = args.output / f"checkpoint-{step + 1}"
            ck.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(ck / "adapter"))
            logger.info("saved checkpoint at step %d to %s", step + 1, ck)

    # Final adapter location matches the RL training output: <output>/adapter
    model.save_pretrained(str(args.output / "adapter"))
    train_log_fh.close()
    logger.info("SFT-only training complete; final adapter at %s/adapter", args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
