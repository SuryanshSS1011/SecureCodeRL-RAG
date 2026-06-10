"""BaselineModel protocol and reference implementations.

The eval harness depends on a narrow `BaselineModel` interface:

    class BaselineModel(Protocol):
        name: str
        def generate(self, prompt: str, *, sampling: SamplingConfig) -> CompletionResult: ...

Two reference implementations:
  - `MockModel` returns canned text per prompt. For unit tests.
  - `HfBaselineModel` wraps a HuggingFace transformers model + tokenizer.
    Stub in v0.1 (raises NotImplementedError); lands when torch + HF are
    installed in the venv.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Callable, Optional, Protocol, runtime_checkable


_FENCE_RE = re.compile(r"```(c|cpp|c\+\+|python|py)?\s*\n(.*?)```", re.DOTALL)
_CWE_RE = re.compile(r"CWE-\d{1,4}")


def _reframe_for_base_model(prompt: str) -> str:
    """Reframe an instruction-style eval prompt as code continuation.

    Base / FIM code models (no chat_template) refuse instruction prose and
    emit EOS as the first generated token, producing empty completions.
    Verified on starcoderbase-3b-safecoder against v0.1.5 eval prompts:
    every prompt produced "".

    The reframing keeps the source block from the eval prompt, drops the
    "Review it and produce an equivalent program. Return ONLY ..." prose,
    and ends with a leading comment cueing the next (hardened) version so
    the model continues with code. If the prompt has no extractable code
    block (rare; some prompts are pure description), it falls back to a
    minimal "# Hardened version that avoids CWE-XXX:" prefix.
    """
    fence = _FENCE_RE.search(prompt)
    cwe_match = _CWE_RE.search(prompt)
    cwe_tag = cwe_match.group(0) if cwe_match else "the named CWE"

    if fence:
        lang_tag = (fence.group(1) or "").lower()
        is_python = lang_tag.startswith("py")
        body = fence.group(2).rstrip()
        if is_python:
            return (
                f"# Original program (must remain free of {cwe_tag}):\n"
                f"{body}\n"
                f"\n"
                f"# Hardened version of the same program, still free of {cwe_tag}:\n"
            )
        return (
            f"/* Original program (must remain free of {cwe_tag}): */\n"
            f"{body}\n"
            f"\n"
            f"/* Hardened version of the same program, still free of {cwe_tag}: */\n"
        )

    # No fenced source. Just cue the model with a comment.
    lower = prompt.lower()
    is_python = "python" in lower or "def " in lower or "import " in lower
    if is_python:
        return f"# {prompt.strip()}\n# Implementation (must avoid {cwe_tag}):\n"
    return f"/* {prompt.strip()} */\n/* Implementation (must avoid {cwe_tag}): */\n"


@dataclass
class SamplingConfig:
    """Decoding configuration. See docs/eval_harness_spec.md §2.3."""

    temperature: float = 0.0
    top_p: float = 1.0
    max_new_tokens: int = 512
    n_samples: int = 1
    seed: int = 42


@dataclass
class CompletionResult:
    """One model output. Token counts are optional; tokenizer may not be
    available for all backends."""

    text: str
    n_input_tokens: Optional[int] = None
    n_output_tokens: Optional[int] = None
    duration_s: float = 0.0
    crashed: bool = False
    metadata: dict = field(default_factory=dict)


@runtime_checkable
class BaselineModel(Protocol):
    """Narrow interface every baseline must implement."""

    name: str

    def generate(
        self, prompt: str, *, sampling: SamplingConfig
    ) -> CompletionResult: ...


class MockModel:
    """A model that returns a pre-supplied completion per prompt.

    `responses` is a callable `prompt -> completion text` (so tests can
    parametrize behavior) OR a dict `prompt -> text` (for explicit
    fixtures). If `prompt` is not in the dict, returns the `default`
    string.
    """

    def __init__(
        self,
        name: str,
        responses: dict[str, str] | Callable[[str], str],
        default: str = "",
        crash_on_prompts: frozenset[str] = frozenset(),
    ) -> None:
        self.name = name
        self._responses = responses
        self._default = default
        self._crash_on = crash_on_prompts

    def generate(
        self, prompt: str, *, sampling: SamplingConfig
    ) -> CompletionResult:
        start = time.monotonic()
        if prompt in self._crash_on:
            return CompletionResult(
                text="",
                duration_s=time.monotonic() - start,
                crashed=True,
            )
        if callable(self._responses):
            text = self._responses(prompt)
        else:
            text = self._responses.get(prompt, self._default)
        return CompletionResult(
            text=text,
            duration_s=time.monotonic() - start,
        )


class HfBaselineModel:
    """Wraps a HuggingFace transformers model for zero-shot generation.

    Construction is cheap: just records the model id. The model and
    tokenizer are loaded lazily on first `generate()` call so the
    registry can be imported without torch installed.

    Prompt formatting uses the tokenizer's chat template when available
    (assuming an `instruct`-style model). Otherwise the prompt is fed in
    raw as a completion-style continuation.
    """

    def __init__(
        self,
        name: str,
        model_id: str,
        device: str = "cuda",
        torch_dtype: str = "bfloat16",
        trust_remote_code: bool = False,
    ) -> None:
        self.name = name
        self.model_id = model_id
        self.device = device
        self.torch_dtype = torch_dtype
        self.trust_remote_code = trust_remote_code
        self._model = None
        self._tokenizer = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise NotImplementedError(
                f"HfBaselineModel requires torch + transformers ({exc}). "
                "Install in the training venv, or use MockModel for unit tests."
            ) from exc

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        dtype = dtype_map.get(self.torch_dtype, torch.bfloat16)

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, trust_remote_code=self.trust_remote_code
        )
        # Many code models leave pad_token unset; reuse eos_token.
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
            trust_remote_code=self.trust_remote_code,
        )
        if self.device == "cuda" and torch.cuda.is_available():
            self._model = self._model.to("cuda")
        else:
            self.device = "cpu"
        self._model.eval()

    def _format_prompt(self, prompt: str) -> str:
        """Format prompt per model kind.

        Instruct models (Qwen, DeepSeek-Coder-Instruct, etc.) carry a chat
        template — use it. Base / FIM code models (StarCoderBase + the
        SafeCoder reproduction on top of it, CodeGen-multi, StarCoder2-base,
        etc.) do not — they were pretrained on raw GitHub code and refuse
        / emit EOS when given instruction prose. For those, the eval prompt
        (which our adapters emit as instruction prose plus a fenced source
        block) is rewritten as a code-continuation: drop the prose, keep
        the source, and frame it as the "original" plus a leading comment
        cueing the "hardened" version. The model then continues writing
        code, which is what it was pretrained to do.
        """
        assert self._tokenizer is not None
        chat_template = getattr(self._tokenizer, "chat_template", None)
        if chat_template:
            messages = [{"role": "user", "content": prompt}]
            return self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        return _reframe_for_base_model(prompt)

    def generate(
        self, prompt: str, *, sampling: SamplingConfig
    ) -> CompletionResult:
        import time

        try:
            self._ensure_loaded()
        except NotImplementedError:
            raise

        import torch  # already verified by _ensure_loaded

        assert self._model is not None and self._tokenizer is not None
        formatted = self._format_prompt(prompt)
        inputs = self._tokenizer(formatted, return_tensors="pt").to(self.device)
        n_input = int(inputs.input_ids.shape[1])

        gen_kwargs = {
            "max_new_tokens": sampling.max_new_tokens,
            "pad_token_id": self._tokenizer.pad_token_id,
        }
        if sampling.temperature > 0.0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = sampling.temperature
            gen_kwargs["top_p"] = sampling.top_p
        else:
            gen_kwargs["do_sample"] = False

        # Seeding when sampling is on. Greedy is deterministic regardless.
        if sampling.temperature > 0.0:
            torch.manual_seed(sampling.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(sampling.seed)

        start = time.monotonic()
        with torch.no_grad():
            output_ids = self._model.generate(**inputs, **gen_kwargs)
        duration = time.monotonic() - start

        # Only the *new* tokens are the completion.
        new_ids = output_ids[0, n_input:]
        text = self._tokenizer.decode(new_ids, skip_special_tokens=True)

        return CompletionResult(
            text=text,
            n_input_tokens=n_input,
            n_output_tokens=int(new_ids.shape[0]),
            duration_s=duration,
        )
