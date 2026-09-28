"""Centralized prompt-format normalizer.

Why: the data adapters historically each rolled their own prompt format.
After the 2026-06-14 audit (FINDINGS_LOG.md), CVEfixes was emitting bare
signatures with no instruction, SecCodePLT only added the "complete the
function" hint conditionally, and SecurityEval / CWEval were pass-through.
The 1.5B instruction-tuned policy responds to bare signatures with prose
or refusals on most rollouts, which crashes `r_reliability` to 0 across
the training pool. With no reliability variance, GRPO has nothing to do.

What this module does: take any Prompt and return a Prompt whose
`prompt_text` is in a single canonical format with:
  1. A short imperative instruction ("Complete the following function..."
     or "Rewrite the following code so that ...").
  2. The signature / description / context the adapter provided.
  3. An explicit code-fence hint in the target language ("```python ...
     ```") with a "respond with ONLY code, no commentary" directive.

The normalizer is idempotent: if the input prompt already looks
well-formed (has an instruction verb in the first 200 chars AND mentions
either a code fence OR the phrase "Return ONLY"), it is returned
unchanged. This means CASTLE / DiverseVul / CyberSecEval prompts that
already meet the standard are passed through verbatim, while CVEfixes /
bare-signature SecCodePLT prompts get wrapped.

The original prompt is preserved in `metadata['prompt_text_pre_normalize']`
so the rebuild is auditable and reversible.
"""

from __future__ import annotations

import dataclasses
import re
from dataclasses import dataclass
from typing import Optional

from .schema import Language, Prompt

__all__ = [
    "NORMALIZER_VERSION",
    "PromptNormalizer",
    "PromptNormalizerConfig",
    "is_well_formed_prompt",
    "normalize_prompt",
]


NORMALIZER_VERSION = "1.0.0"


# Per-language metadata for the code-fence hint.
_FENCE_LANG: dict[Language, str] = {
    Language.PYTHON: "python",
    Language.C: "c",
    Language.CPP: "cpp",
}


# Verbs that count as a valid imperative instruction. Match case-insensitive
# at the head of the prompt.
_INSTRUCTION_VERBS = (
    "complete", "implement", "write", "rewrite", "fix", "patch",
    "produce", "generate", "create", "modify", "refactor", "review",
    "return", "respond",
)

_INSTRUCTION_HEAD_RE = re.compile(
    r"\b(?:" + "|".join(_INSTRUCTION_VERBS) + r")\b", re.IGNORECASE
)

# Markers that indicate the prompt already enforces "code only" output.
_CODE_ONLY_MARKERS = (
    "return only",
    "respond with only",
    "no commentary",
    "no preamble",
    "no markdown fences",
    "only return the code",
    "wrap your code in",
)

_CODE_FENCE_HINT_RE = re.compile(r"```\s*(?:python|py|c|cpp|c\+\+)?\b")


@dataclass(frozen=True)
class PromptNormalizerConfig:
    """Tunables for the normalizer. Defaults match the v0.1.5.2 corpus."""

    # When True, treat prompts that already pass `is_well_formed_prompt` as
    # idempotent and return them unchanged. When False, force the canonical
    # wrapper onto every prompt (useful for ablation runs).
    skip_well_formed: bool = True

    # When True, record the pre-normalization prompt text under
    # metadata['prompt_text_pre_normalize']. Auditability: makes the
    # rebuild reversible.
    record_original: bool = True

    # When True, record per-prompt whether the wrapper was applied under
    # metadata['prompt_normalize_applied']. Useful for downstream analysis
    # of "did fixing the prompt format help on the subset we wrapped".
    record_applied: bool = True


def is_well_formed_prompt(prompt_text: str) -> bool:
    """True if `prompt_text` already has an instruction + code-only directive.

    The check is intentionally loose: we want to leave well-formed prompts
    alone, not enforce one specific phrasing. The two requirements are:
      - The first ~200 chars contain an imperative verb (Complete, Implement,
        Rewrite, Return, ...).
      - The full text contains EITHER a code-only marker ("Return ONLY",
        "Only return the code", "Wrap your code in") OR a code-fence hint
        ("```python", "```c").
    """
    head = prompt_text[:200]
    if not _INSTRUCTION_HEAD_RE.search(head):
        return False
    text_lower = prompt_text.lower()
    has_marker = any(m in text_lower for m in _CODE_ONLY_MARKERS)
    has_fence = bool(_CODE_FENCE_HINT_RE.search(prompt_text))
    return has_marker or has_fence


def _instruction_for(prompt: Prompt) -> str:
    """Pick the instruction verb based on what the adapter gave us.

    If we have a task_signature, the right framing is "complete the
    following function" (sig is the API contract). If we have no signature
    but a textual description, the right framing is "implement the
    following X function ..." (description is the spec).
    """
    lang_label = {
        Language.PYTHON: "Python",
        Language.C: "C",
        Language.CPP: "C++",
    }.get(prompt.language, "code")

    cwe = prompt.target_cwe
    cwe_clause = f" The implementation must be free of {cwe}." if cwe else ""

    if prompt.task_signature:
        return (
            f"Complete the following {lang_label} function.{cwe_clause} "
            f"Return ONLY the complete function body wrapped in a "
            f"```{_FENCE_LANG.get(prompt.language, '')} ... ``` block. "
            f"No commentary, no explanation."
        )
    return (
        f"Implement the following {lang_label} task.{cwe_clause} "
        f"Return ONLY the complete code wrapped in a "
        f"```{_FENCE_LANG.get(prompt.language, '')} ... ``` block. "
        f"No commentary, no explanation."
    )


def _canonical_body(prompt: Prompt) -> str:
    """Render the signature + original prompt body into the canonical form."""
    sig = (prompt.task_signature or "").strip()
    body = (prompt.prompt_text or "").strip()

    # If the body and signature are the same (CVEfixes case: prompt_text =
    # signature), don't duplicate.
    if sig and body == sig:
        return f"Signature:\n```{_FENCE_LANG.get(prompt.language, '')}\n{sig}\n```"

    if sig and body and sig in body:
        # The description already contains the signature.
        return body

    if sig and body:
        return (
            f"Signature:\n```{_FENCE_LANG.get(prompt.language, '')}\n{sig}\n```\n\n"
            f"Description:\n{body}"
        )

    if sig:
        return f"Signature:\n```{_FENCE_LANG.get(prompt.language, '')}\n{sig}\n```"

    return body


def normalize_prompt(
    prompt: Prompt,
    config: Optional[PromptNormalizerConfig] = None,
) -> Prompt:
    """Return a Prompt with `prompt_text` in canonical instruction + code-fence form.

    Idempotent on prompts that already pass `is_well_formed_prompt` (unless
    config.skip_well_formed is False). The original prompt text is stashed
    in metadata when config.record_original is True.
    """
    cfg = config or PromptNormalizerConfig()

    original = prompt.prompt_text
    if cfg.skip_well_formed and is_well_formed_prompt(original):
        if cfg.record_applied:
            new_meta = dict(prompt.metadata or {})
            new_meta["prompt_normalize_applied"] = False
            new_meta["prompt_normalize_version"] = NORMALIZER_VERSION
            return dataclasses.replace(prompt, metadata=new_meta)
        return prompt

    instruction = _instruction_for(prompt)
    body = _canonical_body(prompt)
    new_prompt_text = f"{instruction}\n\n{body}".strip()

    new_meta = dict(prompt.metadata or {})
    if cfg.record_original:
        new_meta["prompt_text_pre_normalize"] = original
    if cfg.record_applied:
        new_meta["prompt_normalize_applied"] = True
        new_meta["prompt_normalize_version"] = NORMALIZER_VERSION

    return dataclasses.replace(
        prompt,
        prompt_text=new_prompt_text,
        metadata=new_meta,
    )


class PromptNormalizer:
    """Stateful wrapper for batch use; collects audit stats.

    Usage:
        norm = PromptNormalizer()
        normalized = [norm(p) for p in prompts]
        print(norm.stats())  # {"total": N, "wrapped": K, "passed_through": N-K, ...}
    """

    def __init__(self, config: Optional[PromptNormalizerConfig] = None) -> None:
        self.config = config or PromptNormalizerConfig()
        self._n_total = 0
        self._n_wrapped = 0
        self._n_passed_through = 0
        self._n_by_source_wrapped: dict[str, int] = {}
        self._n_by_source_total: dict[str, int] = {}

    def __call__(self, prompt: Prompt) -> Prompt:
        self._n_total += 1
        self._n_by_source_total[prompt.source] = (
            self._n_by_source_total.get(prompt.source, 0) + 1
        )

        normalized = normalize_prompt(prompt, self.config)
        applied = (normalized.metadata or {}).get("prompt_normalize_applied", False)
        if applied:
            self._n_wrapped += 1
            self._n_by_source_wrapped[prompt.source] = (
                self._n_by_source_wrapped.get(prompt.source, 0) + 1
            )
        else:
            self._n_passed_through += 1
        return normalized

    def stats(self) -> dict:
        per_source = {}
        for src, total in self._n_by_source_total.items():
            wrapped = self._n_by_source_wrapped.get(src, 0)
            per_source[src] = {
                "total": total,
                "wrapped": wrapped,
                "passed_through": total - wrapped,
                "wrap_rate": (wrapped / total) if total else 0.0,
            }
        return {
            "normalizer_version": NORMALIZER_VERSION,
            "total": self._n_total,
            "wrapped": self._n_wrapped,
            "passed_through": self._n_passed_through,
            "wrap_rate": (self._n_wrapped / self._n_total) if self._n_total else 0.0,
            "per_source": per_source,
        }
