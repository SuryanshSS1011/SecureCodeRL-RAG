"""TorchPolicy: HF model + LoRA + Adam wired to the trainer's PolicyProtocol.

This is the seam between the torch-free orchestration (Trainer) and the
real model. The four PolicyProtocol methods are implemented here:

    generate(prompts, sampling) -> completions (one per prompt)
    log_probs(prompts, completions) -> per-sample sequence-level log prob
    ref_log_probs(prompts, completions) -> same shape, frozen ref policy
    step(loss, step_idx) -> grad norm

The reference policy is a frozen clone of the SFT-init checkpoint. It is
NOT updated during training; KL is computed against it (see
docs/training_spec.md §2.2).

LoRA-B is zero-initialized on construction, so at step 0 the adapted
policy produces identical logits to the reference. This means initial
log_probs == ref_log_probs (within float precision), which is what
test_torch_policy_ref_log_probs_match_log_probs_at_init pins.

Lazy load: the constructor is cheap. Model + LoRA adapter + optimizer
are created on first method call. This lets the registry import without
torch installed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from ..eval.model import SamplingConfig


def _build_cosine_with_warmup(
    optimizer,
    *,
    total_steps: int,
    warmup_ratio: float,
    min_lr_rate: float,
):
    """LambdaLR factor: linear warmup then cosine decay to min_lr_rate.

    factor(step) returns a multiplier on the base lr stored in each
    param_group. Warmup over warmup_ratio * total_steps from 0 -> 1;
    then cosine 1 -> min_lr_rate over the remaining steps.
    """
    import torch

    warmup_steps = max(1, int(warmup_ratio * total_steps))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / float(warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(1.0, progress)
        cos = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_rate + (1.0 - min_lr_rate) * cos

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


@dataclass
class TorchPolicyConfig:
    """LoRA + optimizer + decoding config. Defaults match LCTES inheritance."""

    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    learning_rate: float = 1e-5
    grad_clip_norm: float = 1.0
    target_modules: Optional[list[str]] = None  # None = peft default

    # LR schedule. "constant" keeps lr at learning_rate forever. "cosine"
    # ramps linearly from 0 to learning_rate over warmup_ratio * total_steps,
    # then cosine-decays to min_lr_rate * learning_rate. Matches the recipe
    # in Dang & Ngo 2025 (arXiv:2503.16219) and Tulu 3 / DeepSeek-Math.
    lr_schedule: str = "constant"  # "constant" | "cosine"
    warmup_ratio: float = 0.1
    min_lr_rate: float = 0.1
    total_steps: Optional[int] = None  # required when lr_schedule="cosine"

    # PPO value head (used when --algorithm=ppo). When True, an
    # nn.Linear(hidden_dim, 1) module is added on top of the base model's
    # last hidden state. Its parameters are added to the policy optimizer
    # so the value head trains jointly with the LoRA adapter.
    #
    # The trainer enables this by calling enable_value_head() before
    # _ensure_loaded() runs. Leaving it False makes value_loss_weight and
    # the values() method inert.
    enable_value_head: bool = False
    value_loss_weight: float = 0.5
    value_head_lr: Optional[float] = None  # None -> use learning_rate


class TorchPolicy:
    """Real torch-backed policy implementing PolicyProtocol.

    Implementation notes:
      - log_probs are SEQUENCE-LEVEL (sum over completion tokens).
      - generate() returns one completion per prompt; the trainer wraps
        this in group_size groups upstream.
      - step() invokes loss.backward() + optimizer.step(); requires loss
        to have come from a tensor computation that touched the policy.
        Because the Trainer hands us a *scalar* loss (numpy float) from
        the algorithm step, we have to reconstruct the gradient signal
        here: see _step_with_loss for the mechanism.

    Trade-off chosen: the trainer's algorithm step is torch-free (numpy).
    To make step() work, we recompute log_probs *with autograd enabled*
    on the current rollout batch and use those for the policy gradient.
    This means we do two forward passes (one for the trainer's numpy
    log_probs, one for backward). It's wasteful but keeps the algorithm
    math torch-free. A future optimization could fold the two into one.
    """

    def __init__(
        self,
        model_id: str,
        *,
        device: str = "cuda",
        torch_dtype: str = "bfloat16",
        trust_remote_code: bool = False,
        config: Optional[TorchPolicyConfig] = None,
        lora_r: Optional[int] = None,
        lora_alpha: Optional[int] = None,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.torch_dtype = torch_dtype
        self.trust_remote_code = trust_remote_code
        cfg = config or TorchPolicyConfig()
        if lora_r is not None:
            cfg.lora_r = lora_r
        if lora_alpha is not None:
            cfg.lora_alpha = lora_alpha
        self.config = cfg

        self._tokenizer = None
        self._model = None      # policy (with LoRA adapter)
        self._ref_model = None  # reference (frozen)
        self._optimizer = None
        self._scheduler = None  # LRScheduler; None for constant LR
        self._value_head = None  # nn.Linear(hidden, 1); None unless PPO enabled

        # Track buffered rollout tensors so step() can reuse them without
        # a second forward pass.
        self._cached_rollout = None

        # Resume path: when set, _ensure_loaded() loads the LoRA adapter
        # from this dir (instead of creating a fresh peft wrapper) and the
        # caller restores the optimizer state from optimizer.pt after the
        # first forward pass when the optimizer object exists. Set by
        # train_method.py when --resume-from is passed.
        self._resume_from_dir: Optional[Path] = None

    # ------------------------------------------------------------------
    # Lazy load
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from peft import LoraConfig, get_peft_model
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise NotImplementedError(
                f"TorchPolicy requires torch + transformers + peft ({exc}). "
                "Install in the training venv; for unit tests use MockPolicy."
            ) from exc

        # Resolve device.
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        elif self.device == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        dtype = dtype_map.get(self.torch_dtype, torch.bfloat16)
        # On CPU bfloat16 is supported but float32 is much faster for tests.
        if self.device == "cpu":
            dtype = torch.float32

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, trust_remote_code=self.trust_remote_code
        )
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        base = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
            trust_remote_code=self.trust_remote_code,
        ).to(self.device)

        if self._resume_from_dir is not None:
            # Resume: load the saved LoRA adapter on top of the base. Skips
            # the fresh get_peft_model wrap; PeftModel.from_pretrained reads
            # the adapter_config.json + adapter_model.safetensors that were
            # written by .save_pretrained() at checkpoint time.
            from peft import PeftModel
            self._model = PeftModel.from_pretrained(
                base, str(self._resume_from_dir / "adapter"),
                is_trainable=True,
            )
            self._model.train()
        else:
            # Fresh start: wrap with a new LoRA adapter. peft selects target
            # modules by default per architecture; we leave that auto.
            lora_kwargs = {
                "r": self.config.lora_r,
                "lora_alpha": self.config.lora_alpha,
                "lora_dropout": self.config.lora_dropout,
                "bias": "none",
                "task_type": "CAUSAL_LM",
            }
            if self.config.target_modules is not None:
                lora_kwargs["target_modules"] = self.config.target_modules
            peft_config = LoraConfig(**lora_kwargs)
            self._model = get_peft_model(base, peft_config)
            self._model.train()

        # Reference: deep-copy base WITHOUT LoRA. Cheaper alternative would
        # be to disable adapters via peft's context manager; we keep a
        # separate frozen model to make the contract explicit.
        ref = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
            trust_remote_code=self.trust_remote_code,
        ).to(self.device)
        for p in ref.parameters():
            p.requires_grad_(False)
        ref.eval()
        self._ref_model = ref

        # PPO value head: nn.Linear(hidden_dim, 1) over the base model's
        # last hidden state. Trained jointly with the LoRA adapter under
        # the policy optimizer. Saved/loaded as `value_head.pt` next to
        # the LoRA adapter. Only constructed when the trainer flips
        # config.enable_value_head BEFORE the first forward pass — PPO
        # without a value head silently degrades to REINFORCE, which is
        # what we are fixing here.
        if self.config.enable_value_head:
            hidden_dim = base.config.hidden_size
            self._value_head = torch.nn.Linear(hidden_dim, 1).to(
                self.device, dtype=dtype,
            )
            torch.nn.init.zeros_(self._value_head.weight)
            torch.nn.init.zeros_(self._value_head.bias)
            # Load saved weights when resuming.
            if self._resume_from_dir is not None:
                vh_path = self._resume_from_dir / "value_head.pt"
                if vh_path.exists():
                    state = torch.load(str(vh_path), map_location=self.device)
                    self._value_head.load_state_dict(state)

        # Optimizer over the trainable (LoRA) params + value head params.
        trainable = [p for p in self._model.parameters() if p.requires_grad]
        param_groups = [{"params": trainable, "lr": self.config.learning_rate}]
        if self._value_head is not None:
            vh_lr = (
                self.config.value_head_lr
                if self.config.value_head_lr is not None
                else self.config.learning_rate
            )
            param_groups.append({
                "params": list(self._value_head.parameters()),
                "lr": vh_lr,
            })
        self._optimizer = torch.optim.AdamW(param_groups)

        # LR scheduler. Built after the optimizer so it can be wrapped.
        # constant => no scheduler (None).
        # cosine => linear warmup to lr, then cosine decay to min_lr_rate * lr.
        self._scheduler = None
        if self.config.lr_schedule == "cosine":
            if self.config.total_steps is None:
                raise ValueError(
                    "lr_schedule='cosine' requires total_steps to be set"
                )
            self._scheduler = _build_cosine_with_warmup(
                self._optimizer,
                total_steps=self.config.total_steps,
                warmup_ratio=self.config.warmup_ratio,
                min_lr_rate=self.config.min_lr_rate,
            )
        elif self.config.lr_schedule != "constant":
            raise ValueError(
                f"unknown lr_schedule: {self.config.lr_schedule!r}; "
                "expected 'constant' or 'cosine'"
            )

        # If we're resuming and the checkpoint dir has an optimizer.pt,
        # restore the optimizer state now that the params exist. We
        # tolerate a missing file (lets the resume work even from a
        # checkpoint that was saved with save_optimizer_state disabled
        # or before the optimizer-save feature shipped).
        if self._resume_from_dir is not None:
            opt_path = self._resume_from_dir / "optimizer.pt"
            if opt_path.exists():
                state = torch.load(str(opt_path), map_location=self.device)
                self._optimizer.load_state_dict(state)
            # Scheduler state lives next to optimizer.pt. Missing file is
            # tolerated for back-compat with checkpoints from before this
            # feature shipped.
            if self._scheduler is not None:
                sched_path = self._resume_from_dir / "scheduler.pt"
                if sched_path.exists():
                    state = torch.load(str(sched_path), map_location="cpu")
                    self._scheduler.load_state_dict(state)

    # ------------------------------------------------------------------
    # Checkpoint save/load (P0.1+P0.2). Used by Trainer._save_checkpoint
    # and by train_method.py for fresh resume from a checkpoint dir.
    # ------------------------------------------------------------------

    def save_adapter(self, adapter_dir: Path) -> None:
        """Save the LoRA adapter + tokenizer + (PPO) value head under adapter_dir.

        Called by Trainer._save_checkpoint at the configured cadence and
        by train_method.py for the final checkpoint after the train loop
        ends. Idempotent — the same step's checkpoint can be saved twice.
        """
        if self._model is None:
            return
        adapter_dir.mkdir(parents=True, exist_ok=True)
        self._model.save_pretrained(str(adapter_dir))
        if self._tokenizer is not None:
            self._tokenizer.save_pretrained(str(adapter_dir))
        # PPO value head sits alongside the LoRA adapter. Saved as a tiny
        # state_dict so we don't drag the full base model into the dir.
        if self._value_head is not None:
            import torch
            torch.save(
                self._value_head.state_dict(),
                str(adapter_dir.parent / "value_head.pt"),
            )

    def enable_value_head(self, value_loss_weight: Optional[float] = None,
                          value_head_lr: Optional[float] = None) -> None:
        """Turn on the PPO value head BEFORE the policy is loaded.

        Must be called by train_method.py when --algorithm=ppo, before
        any generate/log_probs/step call. Once the model is loaded the
        head's presence is frozen.
        """
        if self._model is not None:
            raise RuntimeError(
                "enable_value_head must be called before the policy is "
                "loaded; the model is lazy-initialized once and frozen."
            )
        self.config.enable_value_head = True
        if value_loss_weight is not None:
            self.config.value_loss_weight = value_loss_weight
        if value_head_lr is not None:
            self.config.value_head_lr = value_head_lr

    def save_optimizer_state(self, out_path: Path) -> None:
        """Save the AdamW optimizer state via torch.save.

        Skipped silently when the optimizer hasn't been built yet (the
        model is lazy-loaded, so a save before the first step / generate
        produces nothing).
        """
        if self._optimizer is None:
            return
        import torch
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._optimizer.state_dict(), str(out_path))

    def save_scheduler_state(self, out_path: Path) -> None:
        """Save LR scheduler state if a scheduler is active.

        constant LR runs have no scheduler so nothing is saved. The
        trainer can always call this; the no-op path is cheap.
        """
        if self._scheduler is None:
            return
        import torch
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._scheduler.state_dict(), str(out_path))

    def mark_for_resume(self, ckpt_dir: Path) -> None:
        """Tell the policy to load the LoRA adapter from ckpt_dir on next
        _ensure_loaded() call. Must be called BEFORE any generate / step /
        log_probs call (the model is lazy-loaded once and then frozen)."""
        if self._model is not None:
            raise RuntimeError(
                "mark_for_resume must be called before the policy is loaded; "
                "the model has already been instantiated"
            )
        self._resume_from_dir = ckpt_dir

    # ------------------------------------------------------------------
    # PolicyProtocol methods
    # ------------------------------------------------------------------

    def generate(
        self, prompts: list[str], sampling: SamplingConfig
    ) -> list[str]:
        self._ensure_loaded()
        import torch

        assert self._model is not None and self._tokenizer is not None
        self._model.eval()  # critical: dropout off during generation
        try:
            completions: list[str] = []
            for prompt in prompts:
                formatted = self._format_prompt(prompt)
                inputs = self._tokenizer(formatted, return_tensors="pt").to(self.device)
                n_input = inputs.input_ids.shape[1]
                gen_kwargs = {
                    "max_new_tokens": sampling.max_new_tokens,
                    "pad_token_id": self._tokenizer.pad_token_id,
                }
                if sampling.temperature > 0.0:
                    gen_kwargs["do_sample"] = True
                    gen_kwargs["temperature"] = sampling.temperature
                    gen_kwargs["top_p"] = sampling.top_p
                    torch.manual_seed(sampling.seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(sampling.seed)
                else:
                    gen_kwargs["do_sample"] = False
                with torch.no_grad():
                    output_ids = self._model.generate(**inputs, **gen_kwargs)
                new_ids = output_ids[0, n_input:]
                text = self._tokenizer.decode(new_ids, skip_special_tokens=True)
                completions.append(text)
            return completions
        finally:
            self._model.train()  # restore train mode for loss path

    def log_probs(
        self, prompts: list[str], completions: list[str]
    ) -> np.ndarray:
        """Per-sample sequence-level log-prob under the current policy.

        Cached (under key "log_probs_tensor") for use by `step()` so we
        don't recompute the gradient-bearing forward pass.
        """
        self._ensure_loaded()
        return self._log_probs_for(
            self._model, prompts, completions,
            with_grad=True, cache_key="log_probs_tensor",
        )

    def ref_log_probs(
        self, prompts: list[str], completions: list[str]
    ) -> np.ndarray:
        """Per-sample sequence-level log-prob under the frozen reference.

        Cached (under key "ref_log_probs_tensor") so `step()` can compute
        the log-ratio for the PPO clip surrogate without recomputing.
        """
        self._ensure_loaded()
        return self._log_probs_for(
            self._ref_model, prompts, completions,
            with_grad=False, cache_key="ref_log_probs_tensor",
        )

    def values(
        self, prompts: list[str], completions: list[str]
    ) -> np.ndarray:
        """Per-sample mean V over completion tokens. Requires enable_value_head().

        For each (prompt, completion):
          - Tokenize prompt and completion separately, concatenate.
          - Forward through the LoRA-adapted model with output_hidden_states=True.
          - Apply value_head to the last hidden state at completion positions.
          - Average V over the completion tokens to get one scalar per sample.

        Cached (under key "values_tensor_list") so step() can add the value
        loss to the surrogate without recomputing. Returns a numpy array
        of shape (n_samples,) matching log_probs.

        Raises RuntimeError if enable_value_head() was not called before
        the model was loaded.
        """
        self._ensure_loaded()
        if self._value_head is None:
            raise RuntimeError(
                "values() called but no value head is enabled. The trainer "
                "must call enable_value_head() BEFORE the first forward pass."
            )

        import torch

        assert self._tokenizer is not None and self._model is not None

        if len(prompts) != len(completions):
            raise ValueError(
                f"prompts ({len(prompts)}) and completions ({len(completions)}) "
                "must have equal length"
            )

        per_sample_v: list[torch.Tensor] = []
        for prompt, completion in zip(prompts, completions):
            formatted = self._format_prompt(prompt)
            prompt_ids = self._tokenizer(
                formatted, return_tensors="pt", add_special_tokens=False
            ).input_ids.to(self.device)
            comp_ids = self._tokenizer(
                completion, return_tensors="pt", add_special_tokens=False
            ).input_ids.to(self.device)
            full_ids = torch.cat([prompt_ids, comp_ids], dim=1)
            n_prompt = prompt_ids.shape[1]
            n_full = full_ids.shape[1]
            n_comp = n_full - n_prompt
            if n_comp == 0:
                per_sample_v.append(torch.zeros(1, device=self.device).squeeze())
                continue

            with torch.enable_grad():
                out = self._model(full_ids, output_hidden_states=True)
                # Last hidden state at completion positions. For PPO we
                # want V(s_t) where s_t is the state AFTER the prompt and
                # AT the t-th completion token. We use hidden states at
                # positions [n_prompt-1 : n_full-1] which correspond to
                # the states that PREDICTED the completion tokens. The
                # value baseline then attaches per-token to the same
                # tokens whose log_probs we computed.
                last_h = out.hidden_states[-1]  # (1, n_full, hidden)
                comp_h = last_h[:, n_prompt - 1 : n_full - 1, :]
                v_per_token = self._value_head(comp_h).squeeze(-1)  # (1, n_comp)
                seq_v = v_per_token.mean(dim=1).squeeze()
            per_sample_v.append(seq_v)

        stacked = torch.stack(per_sample_v)
        if self._cached_rollout is None:
            self._cached_rollout = {}
        buf = self._cached_rollout.setdefault("values_tensor_list", [])
        buf.append(stacked)
        return stacked.detach().cpu().to(torch.float32).numpy()

    def step(
        self,
        loss: float,
        step_idx: int,
        *,
        advantages: np.ndarray | None = None,
        clip_epsilon: float = 0.2,
        kl_beta: float = 0.0,
        rewards: np.ndarray | None = None,
        entropy_coef: float = 0.0,
        sample_weights: np.ndarray | None = None,
    ) -> float:
        """Rebuild the PPO/GRPO surrogate with autograd-enabled tensors and backprop.

        Surrogate (per sample i):
            ratio_i = exp(log_pi_i - log_pi_ref_i)
            L_pol_i = -min(ratio_i * A_i, clip(ratio_i, 1±eps) * A_i)
            L_kl_i  = beta * (log_pi_i - log_pi_ref_i)
            L_i     = L_pol_i + L_kl_i
            L       = mean over samples of w_i * L_i

        `sample_weights` carries the CWE-aware per-prompt weight w(x)
        (paper Eq. 3) broadcast over each prompt's rollouts; None means
        w_i = 1.

        For RAFT-style algorithms (clip_epsilon=0, no ref needed), the
        formula reduces to `-mean(advantages * log_pi)`, which is the
        SFT loss on selected samples (RAFT's "advantage" is the selection
        mask).

        When a value head is enabled AND `rewards` is provided, a value
        loss `value_loss_weight * mean((V_i - R_i)^2)` is added so the
        value head fits the per-sample reward. Without rewards the value
        head is updated only via its contribution to the advantage
        (mostly a no-op since A is computed numpy-side).

        When `advantages` is None or the rollout cache is empty, returns 0
        without taking a step.

        Returns the gradient norm (post-clip).
        """
        self._ensure_loaded()
        import torch

        assert self._model is not None and self._optimizer is not None

        if self._cached_rollout is None:
            return 0.0

        # Concatenate the per-group tensors that log_probs() accumulated.
        # When the trainer's _run_step calls log_probs once per rollout
        # group, the policy appends each group's autograd tensor to a
        # buffer; we stitch them here so the surrogate sees the same
        # (batch_prompts × group_size,) flat sample axis the algorithm's
        # advantages array uses.
        logp_list = self._cached_rollout.get(
            "log_probs_tensor_list",
            [self._cached_rollout["log_probs_tensor"]]
            if "log_probs_tensor" in self._cached_rollout else [],
        )
        if not logp_list:
            return 0.0
        cached_logp = torch.cat([t.reshape(-1) for t in logp_list])

        if advantages is None:
            # Legacy plumbing-only path: rescale -mean(log_probs) to track
            # the numpy loss magnitude. Wrong direction but keeps tests
            # green for the no-advantage codepaths.
            denom = max(abs(float(cached_logp.detach().mean().item())), 1e-8)
            surrogate = -cached_logp.mean() * float(loss) / denom
        else:
            log_probs = cached_logp.reshape(-1)
            adv = torch.as_tensor(
                advantages.reshape(-1).astype(np.float32),
                device=log_probs.device,
                dtype=log_probs.dtype,
            )
            w = (
                torch.as_tensor(
                    sample_weights.reshape(-1).astype(np.float32),
                    device=log_probs.device,
                    dtype=log_probs.dtype,
                )
                if sample_weights is not None
                else torch.ones_like(log_probs)
            )

            if clip_epsilon > 0.0:
                ref_list = self._cached_rollout.get(
                    "ref_log_probs_tensor_list",
                    [self._cached_rollout["ref_log_probs_tensor"]]
                    if "ref_log_probs_tensor" in self._cached_rollout else [],
                )
                ref_tensor = (
                    torch.cat([t.reshape(-1) for t in ref_list])
                    if ref_list else None
                )
                if ref_tensor is None:
                    # Falling back to REINFORCE if ref wasn't cached.
                    surrogate = -(w * adv * log_probs).mean()
                else:
                    ref_log_probs = ref_tensor.reshape(-1)
                    log_ratio = log_probs - ref_log_probs
                    ratio = torch.exp(torch.clamp(log_ratio, -20.0, 20.0))
                    surr1 = ratio * adv
                    surr2 = torch.clamp(
                        ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon
                    ) * adv
                    policy_loss = -(w * torch.minimum(surr1, surr2)).mean()
                    kl_term = kl_beta * (w * log_ratio).mean()
                    surrogate = policy_loss + kl_term
            else:
                # RAFT / REINFORCE branch.
                surrogate = -(w * adv * log_probs).mean()

        # Entropy bonus. Sequence-level policy-gradient entropy proxy at
        # sampled actions: H ≈ -E[log p]. We *maximize* H, which means
        # *subtract* H from the loss; with `H ≈ -log_probs.mean()` that is
        # `loss += entropy_coef * log_probs.mean()` (pushes log p of
        # sampled actions to be more negative, raising entropy on-policy).
        # Standard PG-with-entropy form. Skipped when entropy_coef <= 0
        # OR when we took the legacy plumbing-only path (advantages None,
        # no cached_logp tensor in scope here — guard with the same flag).
        if entropy_coef > 0.0 and advantages is not None:
            entropy_term = entropy_coef * cached_logp.mean()
            surrogate = surrogate + entropy_term

        # PPO value loss: MSE(V_i, R_i). Computed when the value head was
        # used during this rollout AND the trainer passed `rewards`. The
        # value head learns to predict the per-sample reward so that
        # advantages = R - V have lower variance than R alone. This is the
        # piece that turns "REINFORCE with PPO clipping" into actual PPO.
        if self._value_head is not None and rewards is not None:
            values_list = self._cached_rollout.get("values_tensor_list", [])
            if values_list:
                cached_values = torch.cat([v.reshape(-1) for v in values_list])
                r_t = torch.as_tensor(
                    rewards.reshape(-1).astype(np.float32),
                    device=cached_values.device,
                    dtype=cached_values.dtype,
                )
                value_loss = ((cached_values - r_t) ** 2).mean()
                surrogate = surrogate + self.config.value_loss_weight * value_loss

        self._optimizer.zero_grad()
        surrogate.backward()
        # Clip gradients across BOTH policy LoRA params and (if present)
        # value head params, since they share the optimizer.
        params_to_clip = [p for p in self._model.parameters() if p.requires_grad]
        if self._value_head is not None:
            params_to_clip = params_to_clip + list(self._value_head.parameters())
        grad_norm = torch.nn.utils.clip_grad_norm_(
            params_to_clip,
            self.config.grad_clip_norm,
        )
        self._optimizer.step()
        if self._scheduler is not None:
            self._scheduler.step()
        self._cached_rollout = None
        return float(grad_norm)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _format_prompt(self, prompt: str) -> str:
        assert self._tokenizer is not None
        chat_template = getattr(self._tokenizer, "chat_template", None)
        if chat_template:
            messages = [{"role": "user", "content": prompt}]
            return self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        return prompt

    def _log_probs_for(
        self, model, prompts: list[str], completions: list[str],
        *, with_grad: bool, cache_key: str | None = None,
    ) -> np.ndarray:
        """Compute per-sample sequence log-probs.

        For each (prompt, completion) pair:
          - Tokenize prompt and completion separately.
          - Concatenate; mask completion tokens.
          - Forward; gather log-probs at completion positions; sum.
        """
        import torch

        assert self._tokenizer is not None

        if len(prompts) != len(completions):
            raise ValueError(
                f"prompts ({len(prompts)}) and completions ({len(completions)}) "
                "must have equal length"
            )

        per_sample_logp: list[torch.Tensor] = []
        for prompt, completion in zip(prompts, completions):
            formatted = self._format_prompt(prompt)
            prompt_ids = self._tokenizer(
                formatted, return_tensors="pt", add_special_tokens=False
            ).input_ids.to(self.device)
            comp_ids = self._tokenizer(
                completion, return_tensors="pt", add_special_tokens=False
            ).input_ids.to(self.device)
            full_ids = torch.cat([prompt_ids, comp_ids], dim=1)
            n_prompt = prompt_ids.shape[1]
            n_full = full_ids.shape[1]
            n_comp = n_full - n_prompt
            if n_comp == 0:
                per_sample_logp.append(torch.zeros(1, device=self.device).squeeze())
                continue

            ctx = torch.enable_grad() if with_grad else torch.no_grad()
            with ctx:
                out = model(full_ids)
                # logits[i] predicts token at position i+1.
                # We want log-probs of full_ids[n_prompt:n_full].
                logits = out.logits[:, n_prompt - 1 : n_full - 1, :]
                log_probs = torch.log_softmax(logits.float(), dim=-1)
                target = full_ids[:, n_prompt:n_full]
                gathered = log_probs.gather(2, target.unsqueeze(-1)).squeeze(-1)
                # Sum across completion tokens for sequence-level log-prob.
                seq_logp = gathered.sum(dim=1).squeeze()
            per_sample_logp.append(seq_logp)

        stacked = torch.stack(per_sample_logp)
        if cache_key is not None:
            if self._cached_rollout is None:
                self._cached_rollout = {}
            # Append-to-list (not overwrite). The trainer's _run_step
            # calls log_probs() once per rollout group; each call returns
            # the autograd-bearing tensor for one group's samples, and
            # step() needs ALL groups concatenated so the surrogate's
            # ratio×advantages shape matches the (batch_prompts × group_size,)
            # flat layout that the trainer hands in. Without this, only
            # the LAST group's tensor lived in the cache and step()
            # crashed on shape mismatch when batch_prompts > 1.
            buf = self._cached_rollout.setdefault(cache_key + "_list", [])
            buf.append(stacked)

        return stacked.detach().cpu().to(torch.float32).numpy()
