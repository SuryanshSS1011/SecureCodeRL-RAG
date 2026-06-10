"""Tests for the centralized prompt-format normalizer.

Backs the 2026-06-14 FINDINGS_LOG bug-fix: CVEfixes and partial-SecCodePLT
prompts were bare signatures, producing prose / refusal completions on the
1.5B policy and ultimately r_reliability=0 across the training pool. This
normalizer wraps them with an instruction + code-fence directive.
"""

from __future__ import annotations

from secure_code_rl_ictai.data_prep.prompt_template import (
    NORMALIZER_VERSION,
    PromptNormalizer,
    PromptNormalizerConfig,
    is_well_formed_prompt,
    normalize_prompt,
)
from secure_code_rl_ictai.data_prep.schema import Language, Prompt, TestSpec


def _mk_prompt(prompt_text: str, signature: str = "", language: Language = Language.PYTHON,
               source: str = "cvefixes", cwe: str = "CWE-89") -> Prompt:
    return Prompt(
        id=f"test::{source}::{hash(prompt_text) & 0xffff}",
        source=source,
        language=language,
        target_cwe=cwe,
        prompt_text=prompt_text,
        test_spec=TestSpec(language=language, test_cases=[]),
        task_signature=signature or None,
        metadata={},
    )


# ---- is_well_formed_prompt ----

class TestIsWellFormedPrompt:
    def test_castle_style_passes(self):
        """CASTLE produces description + instruction + fenced code; should pass."""
        text = (
            "A C program demonstrating CWE-119.\n\n"
            "Below is a C program that contains a CWE-119 vulnerability. "
            "Rewrite it so that the vulnerability is fixed. Return ONLY the "
            "corrected C program; no commentary, no markdown fences.\n\n"
            "```c\nint main() { return 0; }\n```"
        )
        assert is_well_formed_prompt(text) is True

    def test_diversevul_style_passes(self):
        text = (
            "Implement the following C function such that it is free of CWE-787 "
            "(out-of-bounds write). Return ONLY the complete function definition; "
            "no commentary, no markdown fences. Project: linux. Signature: void foo()."
        )
        assert is_well_formed_prompt(text) is True

    def test_cyberseceval_style_passes(self):
        text = (
            "Complete the following C function in a security-conscious way.\n"
            "Only return the code, don't include any other information."
        )
        assert is_well_formed_prompt(text) is True

    def test_bare_signature_fails(self):
        """CVEfixes-style bare signature has neither instruction nor fence hint."""
        assert is_well_formed_prompt("deserialize( self , content , format = 'json' )") is False

    def test_bare_function_name_fails(self):
        assert is_well_formed_prompt("test_query_customer_user( a, b, c )") is False

    def test_instruction_alone_without_code_directive_fails(self):
        """'Complete the function' alone with no fence hint or 'Return ONLY' fails."""
        assert is_well_formed_prompt("Complete the function foo.") is False

    def test_fence_hint_without_instruction_fails(self):
        """A leading ```c with no imperative verb fails."""
        assert is_well_formed_prompt("```c\nint main() {}\n```") is False


# ---- normalize_prompt ----

class TestNormalizePrompt:
    def test_cvefixes_bare_signature_gets_wrapped(self):
        p = _mk_prompt(
            prompt_text="deserialize( self , content , format = 'json' )",
            signature="deserialize( self , content , format = 'json' )",
            source="cvefixes",
        )
        n = normalize_prompt(p)
        assert n.prompt_text != p.prompt_text
        assert "Complete the following Python function" in n.prompt_text
        assert "```python" in n.prompt_text
        assert "Return ONLY" in n.prompt_text
        assert n.metadata["prompt_normalize_applied"] is True
        assert n.metadata["prompt_text_pre_normalize"] == p.prompt_text
        assert n.metadata["prompt_normalize_version"] == NORMALIZER_VERSION

    def test_castle_passes_through(self):
        original_text = (
            "A C program demonstrating CWE-119.\n\n"
            "Below is a C program. Rewrite it. Return ONLY the corrected C program.\n\n"
            "```c\nint main() {}\n```"
        )
        p = _mk_prompt(prompt_text=original_text, source="castle",
                       language=Language.C, cwe="CWE-119")
        n = normalize_prompt(p)
        assert n.prompt_text == original_text
        assert n.metadata["prompt_normalize_applied"] is False

    def test_cwe_included_in_instruction(self):
        p = _mk_prompt(
            prompt_text="bar(int x)",
            signature="bar(int x)",
            source="cvefixes",
            language=Language.C,
            cwe="CWE-787",
        )
        n = normalize_prompt(p)
        assert "CWE-787" in n.prompt_text
        assert "free of CWE-787" in n.prompt_text

    def test_c_uses_c_fence(self):
        p = _mk_prompt(prompt_text="foo(int x)", signature="foo(int x)",
                       language=Language.C, source="cvefixes")
        n = normalize_prompt(p)
        assert "```c" in n.prompt_text and "```cpp" not in n.prompt_text

    def test_cpp_uses_cpp_fence(self):
        p = _mk_prompt(prompt_text="Foo::bar()", signature="Foo::bar()",
                       language=Language.CPP, source="cvefixes", cwe="CWE-119")
        n = normalize_prompt(p)
        assert "```cpp" in n.prompt_text

    def test_signature_and_body_when_equal_not_duplicated(self):
        """CVEfixes case: prompt_text == signature. Don't render the signature twice."""
        sig = "do_template( self , data )"
        p = _mk_prompt(prompt_text=sig, signature=sig, source="cvefixes")
        n = normalize_prompt(p)
        assert n.prompt_text.count(sig) == 1

    def test_signature_in_body_no_extra_render(self):
        """If description already contains the signature, don't add a Signature: section."""
        body = "Function `foo(int x)` should validate the input."
        p = _mk_prompt(prompt_text=body, signature="foo(int x)", source="cvefixes")
        n = normalize_prompt(p)
        assert "Signature:" not in n.prompt_text
        assert "foo(int x)" in n.prompt_text

    def test_idempotent(self):
        """Normalizing a well-formed prompt twice yields the same text."""
        p = _mk_prompt(prompt_text="x()", signature="x()", source="cvefixes")
        n1 = normalize_prompt(p)
        n1_again = normalize_prompt(n1)
        assert n1_again.prompt_text == n1.prompt_text

    def test_skip_well_formed_false_forces_rewrap(self):
        """With skip_well_formed=False, even already-good prompts get re-wrapped."""
        p = _mk_prompt(
            prompt_text="Complete the following. Return ONLY code. ```python\nfoo()\n```",
            source="other",
        )
        forced = normalize_prompt(p, PromptNormalizerConfig(skip_well_formed=False))
        assert forced.metadata["prompt_normalize_applied"] is True

    def test_record_original_disabled(self):
        p = _mk_prompt(prompt_text="x()", signature="x()", source="cvefixes")
        n = normalize_prompt(p, PromptNormalizerConfig(record_original=False))
        assert "prompt_text_pre_normalize" not in n.metadata


# ---- PromptNormalizer stateful wrapper ----

class TestPromptNormalizerStats:
    def test_collects_per_source_stats(self):
        norm = PromptNormalizer()
        prompts = [
            _mk_prompt("foo()", "foo()", source="cvefixes"),
            _mk_prompt("bar()", "bar()", source="cvefixes"),
            _mk_prompt(
                "Implement this. Return ONLY code. ```c\nint x;\n```",
                source="diversevul",
                language=Language.C,
            ),
        ]
        for p in prompts:
            norm(p)
        stats = norm.stats()
        assert stats["total"] == 3
        assert stats["wrapped"] == 2
        assert stats["passed_through"] == 1
        assert stats["per_source"]["cvefixes"]["wrapped"] == 2
        assert stats["per_source"]["diversevul"]["wrapped"] == 0
        assert stats["normalizer_version"] == NORMALIZER_VERSION

    def test_empty_stats(self):
        norm = PromptNormalizer()
        stats = norm.stats()
        assert stats["total"] == 0
        assert stats["wrap_rate"] == 0.0


# ---- end-to-end shape ----

def test_normalizer_preserves_all_other_fields():
    p = _mk_prompt(
        prompt_text="foo()",
        signature="foo()",
        source="cvefixes",
        cwe="CWE-89",
        language=Language.PYTHON,
    )
    p.metadata["arbitrary"] = "preserved"
    n = normalize_prompt(p)
    assert n.id == p.id
    assert n.source == p.source
    assert n.target_cwe == p.target_cwe
    assert n.language == p.language
    assert n.test_spec is p.test_spec
    assert n.task_signature == p.task_signature
    assert n.metadata["arbitrary"] == "preserved"
