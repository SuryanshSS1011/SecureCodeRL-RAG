"""Headline eval entry point for trained LoRA checkpoints.

Loads the base model plus a LoRA adapter (or a full fine-tune) from a
training-run checkpoint, then runs the standard EvalHarness on the v0.1.7
eval set (1,582 prompts) to produce aggregate.json. The output schema
matches the baseline sweep.

Usage:
    PYTHONPATH=src .venv/bin/python scripts/headline_eval.py \\
        --checkpoint /scratch/.../arm_a/reweight/checkpoint-best \\
        --name arm_a_reweight \\
        --eval-jsonl /scratch/.../build/v0.1.5/eval_prompts.jsonl \\
        --output /scratch/.../sweeps/phase1_v0_1_5_headline/arm_a_reweight \\
        --temperature 0.0 --max-new-tokens 512

The checkpoint should be a directory containing `adapter/` (the LoRA
adapter saved by train_method.py at checkpoint-best). Base model defaults
to Qwen2.5-Coder-1.5B-Instruct.

To run inference-time RAG (cell 4 of P1.3), pass --inference-rag-on
--rag-index-dir. The retriever + embedder are loaded inside the
EvalHarness's prompt_transform.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def _build_real_pipeline(compile_mode: str = "syntax_only"):
    """Real RewardPipeline w/ all 4 SAST tools + RealOracle, no RAG.

    Inference-time RAG affects what reaches the generator; the reward
    pipeline at eval time still scores the same way as the baselines did.

    `compile_mode` selects how the oracle judges C/C++ Compile@1:
    "syntax_only" (paper Section V.B: parse for Python, `-fsyntax-only`
    for C/C++; the definition behind every reported Compile@1) or
    "link_and_run" (the training-time reward path, which additionally
    links and executes the binary).
    """
    from secure_code_rl_ictai.reward import (
        RewardCalculator, RewardConfig, RewardPipeline,
    )
    from secure_code_rl_ictai.reward.reliability_oracle import RealOracle
    from secure_code_rl_ictai.sast import ToolName
    from secure_code_rl_ictai.sast.adapters import (
        BanditAdapter, CodeQLAdapter, CppcheckAdapter, SemgrepAdapter,
    )
    from secure_code_rl_ictai.sast.normalizer import SarifNormalizer
    from secure_code_rl_ictai.sast.runner import SastRunner
    from secure_code_rl_ictai.sast.severity import SeveritySource

    def _resolve(name: str) -> str:
        # Resolve absolute path of CLI tools inside the venv.
        candidate = Path(sys.executable).parent / name
        if candidate.exists():
            return str(candidate)
        import shutil
        return shutil.which(name) or name

    bandit = BanditAdapter(bandit_binary=_resolve("bandit"))
    semgrep = SemgrepAdapter(semgrep_binary=_resolve("semgrep"))

    codeql_bin = os.environ.get("CODEQL_BINARY") or _resolve("codeql")
    codeql_at_oss = "/storage/home/sss6371/work/oss/codeql-cli/codeql/codeql"
    if codeql_bin == "codeql" and Path(codeql_at_oss).exists():
        codeql_bin = codeql_at_oss
    codeql = CodeQLAdapter(codeql_binary=codeql_bin)

    cppcheck_bin = os.environ.get("CPPCHECK_BINARY") or _resolve("cppcheck")
    cppcheck_at_build = "/storage/home/sss6371/work/cppcheck_build/cppcheck-2.18.0/cppcheck"
    if cppcheck_bin == "cppcheck" and Path(cppcheck_at_build).exists():
        cppcheck_bin = cppcheck_at_build
    cppcheck = CppcheckAdapter(cppcheck_binary=cppcheck_bin)

    runner = SastRunner(
        adapters={
            ToolName.BANDIT: bandit,
            ToolName.SEMGREP: semgrep,
            ToolName.CODEQL: codeql,
            ToolName.CPPCHECK: cppcheck,
        },
        normalizer=SarifNormalizer(),
    )

    # Resolve C/C++ compilers per host (matches train_method.py).
    import shutil
    c_cc = _resolve("clang") if shutil.which("clang") else _resolve("gcc")
    cpp_cc = _resolve("clang++") if shutil.which("clang++") else _resolve("g++")
    oracle = RealOracle(
        python_executable=_resolve("python3"),
        c_compiler=c_cc,
        cpp_compiler=cpp_cc,
        compile_mode=compile_mode,
    )

    sev_src = SeveritySource(Path("data/nvdlib_cwe_medians.json"))
    return RewardPipeline(
        oracle=oracle,
        sast_runner=runner,
        severity_source=sev_src,
        calculator=RewardCalculator(RewardConfig(), sev_src),
    )


class _LoraBaselineModel:
    """BaselineModel-protocol wrapper around base Qwen + LoRA adapter.

    Also handles full-FT checkpoints: when the checkpoint dir contains a
    full HF model (config.json + model.safetensors / pytorch_model.bin) but
    no adapter_config.json, we load it directly with AutoModelForCausalLM
    and skip the PEFT wrapper. This lets the full-FT cell go through the
    same eval path as the LoRA cells.
    """

    def __init__(
        self,
        name: str,
        adapter_path: Path,
        model_id: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        device: str = "cuda",
        torch_dtype: str = "bfloat16",
    ) -> None:
        self.name = name
        self.adapter_path = adapter_path
        self.model_id = model_id
        self.device = device
        self.torch_dtype = torch_dtype
        self._model = None
        self._tokenizer = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        dtype = dtype_map.get(self.torch_dtype, torch.bfloat16)

        # Detect full-FT vs LoRA from the checkpoint contents.
        is_lora = (self.adapter_path / "adapter_config.json").exists()
        print(f"[headline-eval] loading base {self.model_id} ... (mode="
              f"{'lora' if is_lora else 'full-ft'})",
              file=sys.stderr, flush=True)
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, padding_side="left"
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        if is_lora:
            from peft import PeftModel
            base = AutoModelForCausalLM.from_pretrained(
                self.model_id, torch_dtype=dtype,
            ).to(self.device)
            print(f"[headline-eval] loading LoRA adapter from {self.adapter_path} ...",
                  file=sys.stderr, flush=True)
            self._model = PeftModel.from_pretrained(base, str(self.adapter_path))
        else:
            # Full-FT: load the saved weights directly. Tokenizer still
            # comes from the base model id since save_pretrained on the
            # PEFT-wrapped (or bare) model does not always include the
            # tokenizer files.
            print(f"[headline-eval] loading full-FT weights from {self.adapter_path} ...",
                  file=sys.stderr, flush=True)
            self._model = AutoModelForCausalLM.from_pretrained(
                str(self.adapter_path), torch_dtype=dtype,
            ).to(self.device)
        self._model.eval()
        print(f"[headline-eval] model loaded on {self.device}",
              file=sys.stderr, flush=True)

    def generate(self, prompt: str, *, sampling):
        from secure_code_rl_ictai.eval.model import CompletionResult
        import torch
        self._ensure_loaded()
        # Format with chat template like training did, so apples-to-apples.
        chat_template = getattr(self._tokenizer, "chat_template", None)
        if chat_template:
            messages = [{"role": "user", "content": prompt}]
            formatted = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            formatted = prompt
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
        t0 = time.monotonic()
        with torch.no_grad():
            output_ids = self._model.generate(**inputs, **gen_kwargs)
        wall = time.monotonic() - t0
        new_ids = output_ids[0, n_input:]
        text = self._tokenizer.decode(new_ids, skip_special_tokens=True)
        # CompletionResult fields: text, n_input_tokens, n_output_tokens,
        # duration_s, crashed, metadata. Old code passed wall_s/ok which
        # don't exist.
        return CompletionResult(
            text=text,
            n_input_tokens=n_input,
            n_output_tokens=int(new_ids.shape[0]),
            duration_s=wall,
            crashed=False,
        )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True,
                   help="Path to <output>/checkpoint-best or checkpoint-N. "
                        "Must contain an 'adapter/' subdir with the LoRA.")
    p.add_argument("--name", type=str, required=True,
                   help="Label for the model in output (e.g. 'arm_a_reweight').")
    p.add_argument("--eval-jsonl", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True,
                   help="Output directory. Aggregate written to "
                        "<output>/<name>/aggregate.json + per_prompt_stream.jsonl.")
    p.add_argument("--model-id", type=str,
                   default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--torch-dtype", type=str, default="bfloat16")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-prompts", type=int, default=0,
                   help="0 = all prompts. Set >0 for a smoke pass.")
    # Inference-time RAG control (Section VI-C).
    p.add_argument("--inference-rag-on", action="store_true")
    p.add_argument("--rag-index-dir", type=Path, default=None)
    p.add_argument("--rag-prepend-template", type=str, default=None)
    p.add_argument("--rag-embed-device", type=str, default="cuda")
    # Resume from a partial per_prompt_stream.jsonl. Skips already-completed
    # prompt ids and appends new ones. Use this when a SLURM walltime kill
    # left a stream file behind that you want to continue, instead of
    # starting fresh and re-spending compute on the first N prompts.
    p.add_argument("--resume", action="store_true",
                   help="resume from output_dir/<name>/per_prompt_stream.jsonl if it exists")
    p.add_argument("--compile-mode", choices=("syntax_only", "link_and_run"),
                   default="syntax_only",
                   help="C/C++ Compile@1 judgement. syntax_only = gcc "
                        "-fsyntax-only (the paper's metric definition). "
                        "link_and_run = link + execute (training reward path).")
    args = p.parse_args()

    adapter_dir = args.checkpoint / "adapter"
    if not adapter_dir.exists():
        # Some training paths save LoRA directly into the checkpoint dir.
        adapter_dir = args.checkpoint
    # Accept either LoRA (adapter_config.json) OR full-FT (config.json +
    # weights) checkpoints. The model loader picks the right path.
    has_lora = (adapter_dir / "adapter_config.json").exists()
    has_fullft = (adapter_dir / "config.json").exists()
    if not (has_lora or has_fullft):
        raise FileNotFoundError(
            f"No model checkpoint at {adapter_dir}: expected either "
            f"adapter_config.json (LoRA) or config.json (full-FT)"
        )

    from secure_code_rl_ictai.eval.harness import EvalHarness
    from secure_code_rl_ictai.eval.model import SamplingConfig
    from secure_code_rl_ictai.data_prep.schema import Prompt
    from secure_code_rl_ictai.data_prep import normalize_language, normalize_cwe
    from secure_code_rl_ictai.reward.reliability_oracle import TestCase, TestSpec

    # Load prompts.
    prompts: list[Prompt] = []
    with args.eval_jsonl.open() as fh:
        for line in fh:
            r = json.loads(line)
            lang = normalize_language(r["language"])
            cwe = normalize_cwe(r["target_cwe"])
            ts = r.get("test_spec") or {}
            test_spec = TestSpec(
                language=lang,
                test_cases=[
                    TestCase(
                        input_stdin=str(tc.get("input_stdin", "")),
                        expected_stdout=str(tc.get("expected_stdout", "")),
                        timeout_s=float(tc.get("timeout_s", 5.0)),
                    )
                    for tc in (ts.get("test_cases") or [])
                ],
                extra_files=dict(ts.get("extra_files") or {}),
                compile_flags=list(ts.get("compile_flags", []) or []),
                entry_module=ts.get("entry_module"),
                prefix_text=ts.get("prefix_text"),
                suffix_text=ts.get("suffix_text"),
            )
            prompts.append(Prompt(
                id=r["id"],
                source=r.get("source", "eval"),
                language=lang,
                target_cwe=cwe,
                prompt_text=r["prompt_text"],
                task_signature=r.get("task_signature", r["prompt_text"]),
                test_spec=test_spec,
                metadata=r.get("metadata", {}),
            ))
    if args.max_prompts > 0:
        prompts = prompts[: args.max_prompts]
    print(f"[headline-eval] loaded {len(prompts)} prompts", file=sys.stderr, flush=True)

    # Build the trained LoRA model + reward pipeline.
    model = _LoraBaselineModel(
        name=args.name,
        adapter_path=adapter_dir,
        model_id=args.model_id,
        device=args.device,
        torch_dtype=args.torch_dtype,
    )
    pipeline = _build_real_pipeline(compile_mode=args.compile_mode)

    # Optional inference-time RAG prepend transform.
    prompt_transform = None
    if args.inference_rag_on:
        if args.rag_index_dir is None:
            raise ValueError("--inference-rag-on requires --rag-index-dir")
        from secure_code_rl_ictai.rag import load_embedder, load_retriever
        from secure_code_rl_ictai.rag.retriever import RetrievalQuery
        print(f"[headline-eval] inference-RAG: loading index from {args.rag_index_dir}",
              file=sys.stderr, flush=True)
        retriever = load_retriever(args.rag_index_dir)
        embedder = load_embedder(args.rag_index_dir / "manifest.json",
                                 device=args.rag_embed_device)
        tpl = args.rag_prepend_template or (
            "// Reference secure implementation for {cwe}:\n"
            "{exemplar}\n\n"
            "// Now complete:\n{prompt}"
        )

        def prompt_transform(prompt: Prompt) -> tuple[str, dict]:
            src = prompt.task_signature or prompt.prompt_text
            try:
                qe = embedder.embed(src)
            except Exception as e:
                return prompt.prompt_text, {
                    "rag_used": False, "rag_miss_reason": f"embed_error: {e}",
                }
            q = RetrievalQuery(
                cwe=prompt.target_cwe, task_signature=src,
                query_embedding=qe, top_k_per_backend=20,
            )
            hit = retriever.retrieve(q)
            if hit is None:
                return prompt.prompt_text, {
                    "rag_used": False, "rag_miss_reason": "no_cwe_match",
                }
            wrapped = tpl.format(
                cwe=prompt.target_cwe,
                exemplar=hit.pair.e_pos,
                prompt=prompt.prompt_text,
            )
            return wrapped, {
                "rag_used": True,
                "rag_hit_pair_id": (hit.pair.cve_id
                                     or (hit.pair.task_signature or "")[:64]),
                "rag_score": float(hit.rrf_score),
                "rag_hit_cwe": hit.pair.cwe,
            }

    # Output paths matching baseline-sweep layout: <output>/<name>/aggregate.json
    out_dir = args.output / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Run the harness. Pattern mirrors scripts/run_baseline_sweep.py:
    # EvalHarness(pipeline, sampling) at construction, model is passed
    # to .evaluate(). report.save(output_dir) writes per_prompt.jsonl
    # and aggregate.json under output_dir/<model.name>/.
    sampling = SamplingConfig(
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        n_samples=1,
        seed=args.seed,
    )
    harness = EvalHarness(pipeline=pipeline, sampling=sampling)

    stream_path = out_dir / "per_prompt_stream.jsonl"
    # Pass checkpoint_output_dir so the harness flushes a partial
    # aggregate.json every 200 prompts. Without this, a SLURM walltime
    # kill at prompt N<1529 leaves only the per_prompt_stream with no
    # aggregated metrics file for the paper-table consumer.
    report = harness.evaluate(
        model=model,
        prompts=prompts,
        stream_path=stream_path,
        resume=args.resume,
        prompt_transform=prompt_transform,
        checkpoint_output_dir=args.output,
        checkpoint_every=200,
    )
    # args.output is the parent; report.save writes args.output / model.name /
    # aggregate.json + per_prompt.jsonl. model.name was set to args.name above
    # so the path matches what callers (queue_v0_1_7_headline_eval.sh) expect.
    report.save(args.output)
    print(
        f"[headline-eval] wrote {args.output}/{args.name}/aggregate.json",
        file=sys.stderr, flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
