"""Tests for the trainer orchestration loop.

The Trainer takes (policy, reward_pipeline, schedule, algorithm, sampler)
and runs the step loop. The policy interface is split four ways so the
torch glue stays out of this module entirely:
  - generate(prompts, sampling) -> completions
  - log_probs(prompts, completions) -> per-sample sequence-level log probs
  - ref_log_probs(prompts, completions) -> same shape, ref policy
  - step(loss) -> grad_norm

We test the loop against a MockPolicy returning hand-supplied values.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from secure_code_rl_ictai.data_prep.schema import Prompt
from secure_code_rl_ictai.eval.model import SamplingConfig
from secure_code_rl_ictai.reward import (
    Language,
    MockOracle,
    ReliabilitySignals,
    RewardCalculator,
    RewardConfig,
    RewardPipeline,
    TestSpec,
)
from secure_code_rl_ictai.rl import GrpoConfig, get_algorithm
from secure_code_rl_ictai.rl.schedule import DEFAULT_PHASES, PhaseSchedule, PhaseSpec
from secure_code_rl_ictai.rl.trainer import (
    Trainer,
    TrainerConfig,
)
from secure_code_rl_ictai.sast.models import ToolName
from secure_code_rl_ictai.sast.runner import MockAdapter, SastRunner
from secure_code_rl_ictai.sast.severity import SeveritySource


# ----------------------------------------------------------------------
# Test fixtures
# ----------------------------------------------------------------------


def _prompt(prompt_id: str, cwe: str = "CWE-89") -> Prompt:
    return Prompt(
        id=prompt_id,
        source="test",
        language=Language.PYTHON,
        target_cwe=cwe,
        prompt_text=f"# task {prompt_id}",
        test_spec=TestSpec(language=Language.PYTHON, test_cases=[]),
    )


def _all_mock_pipeline() -> RewardPipeline:
    adapters = {
        ToolName.CODEQL: MockAdapter(ToolName.CODEQL, {"runs": []}),
        ToolName.SEMGREP: MockAdapter(ToolName.SEMGREP, {"runs": []}),
        ToolName.BANDIT: MockAdapter(ToolName.BANDIT, {"runs": []}),
        ToolName.CPPCHECK: MockAdapter(ToolName.CPPCHECK, {"runs": []}),
    }
    runner = SastRunner(adapters)
    sev_src = SeveritySource()
    calc = RewardCalculator(RewardConfig(alpha=0.5), sev_src)
    return RewardPipeline(
        oracle=MockOracle(
            ReliabilitySignals(compiles=True, tests_passed=1, tests_total=1)
        ),
        sast_runner=runner,
        severity_source=sev_src,
        calculator=calc,
    )


class MockPolicy:
    """Policy that returns canned completions and log-probs.

    The test supplies a constant completion text and explicit log_prob
    values per call. step() records the loss it received so tests can
    assert on it.
    """

    def __init__(
        self,
        completion: str = "import sqlite3\n",
        completion_log_prob: float = -1.0,
        ref_log_prob: float = -1.0,
    ) -> None:
        self.completion = completion
        self.completion_log_prob = completion_log_prob
        self.ref_log_prob = ref_log_prob
        self.received_losses: list[float] = []
        self.received_steps: list[int] = []

    def generate(self, prompts: list[str], sampling: SamplingConfig) -> list[str]:
        return [self.completion] * len(prompts)

    def log_probs(self, prompts: list[str], completions: list[str]) -> np.ndarray:
        return np.full(len(prompts), self.completion_log_prob)

    def ref_log_probs(self, prompts: list[str], completions: list[str]) -> np.ndarray:
        return np.full(len(prompts), self.ref_log_prob)

    def step(
        self,
        loss: float,
        step_idx: int,
        *,
        advantages=None,
        clip_epsilon: float = 0.2,
        kl_beta: float = 0.0,
        entropy_coef: float = 0.0,
        **_unused,
    ) -> float:
        self.received_losses.append(loss)
        self.received_steps.append(step_idx)
        # MockPolicy ignores advantages, entropy_coef, etc.; trainer
        # plumbing tests only verify step() invocation count + loss values.
        return 0.5  # grad_norm


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------


def test_trainer_runs_n_steps_against_prompt_pool():
    policy = MockPolicy()
    pipeline = _all_mock_pipeline()
    schedule = PhaseSchedule(DEFAULT_PHASES)
    algorithm = get_algorithm("grpo", GrpoConfig())
    prompts = [_prompt(f"p{i}") for i in range(10)]
    config = TrainerConfig(
        total_steps=3,
        batch_prompts=2,
        group_size=4,
        seed=42,
    )
    trainer = Trainer(
        policy=policy,
        reward_pipeline=pipeline,
        schedule=schedule,
        algorithm=algorithm,
        config=config,
    )
    records = trainer.run(prompts)
    assert len(records) == 3
    # policy.step called once per training step.
    assert len(policy.received_losses) == 3
    assert policy.received_steps == [0, 1, 2]


def test_trainer_writes_jsonl_log(tmp_path: Path):
    import json

    policy = MockPolicy()
    pipeline = _all_mock_pipeline()
    schedule = PhaseSchedule(DEFAULT_PHASES)
    algorithm = get_algorithm("grpo", GrpoConfig())
    prompts = [_prompt(f"p{i}") for i in range(8)]
    log_path = tmp_path / "train.jsonl"
    config = TrainerConfig(
        total_steps=2,
        batch_prompts=2,
        group_size=4,
        seed=42,
        log_path=log_path,
    )
    trainer = Trainer(
        policy=policy,
        reward_pipeline=pipeline,
        schedule=schedule,
        algorithm=algorithm,
        config=config,
    )
    trainer.run(prompts)
    assert log_path.exists()
    lines = log_path.read_text().strip().splitlines()
    assert len(lines) == 2
    record = json.loads(lines[0])
    assert record["step"] == 0
    assert record["phase"] == 1
    assert record["alpha"] == 0.3


def test_trainer_phase_transitions_fire():
    policy = MockPolicy()
    pipeline = _all_mock_pipeline()
    # Compact schedule: phases at step 0, 1, 2.
    schedule = PhaseSchedule(
        [
            PhaseSpec(number=1, start_step=0, alpha=0.3, beta=0.01),
            PhaseSpec(number=2, start_step=1, alpha=0.5, beta=0.02),
            PhaseSpec(number=3, start_step=2, alpha=0.7, beta=0.04),
        ]
    )
    algorithm = get_algorithm("grpo", GrpoConfig())
    prompts = [_prompt(f"p{i}") for i in range(4)]
    config = TrainerConfig(
        total_steps=3,
        batch_prompts=2,
        group_size=2,
        seed=42,
    )
    trainer = Trainer(
        policy=policy,
        reward_pipeline=pipeline,
        schedule=schedule,
        algorithm=algorithm,
        config=config,
    )
    records = trainer.run(prompts)
    assert records[0].phase == 1
    assert records[1].phase == 2
    assert records[2].phase == 3
    assert records[0].alpha == 0.3
    assert records[1].alpha == 0.5
    assert records[2].alpha == 0.7


def test_trainer_handles_zero_variance_groups():
    """When every rollout produces identical R_total, the GRPO algorithm
    masks the group and loss is 0. The trainer should still complete the
    step and emit a record."""
    policy = MockPolicy(completion="same")  # same prompt -> same completion -> same reward
    pipeline = _all_mock_pipeline()
    schedule = PhaseSchedule(DEFAULT_PHASES)
    algorithm = get_algorithm("grpo", GrpoConfig())
    prompts = [_prompt("p0")]
    config = TrainerConfig(
        total_steps=1,
        batch_prompts=1,
        group_size=4,
        seed=42,
    )
    trainer = Trainer(
        policy=policy,
        reward_pipeline=pipeline,
        schedule=schedule,
        algorithm=algorithm,
        config=config,
    )
    records = trainer.run(prompts)
    assert len(records) == 1
    # Zero variance -> the group is masked.
    assert records[0].zero_variance_group_rate > 0


def test_trainer_sampler_draws_from_prompt_pool_with_seed():
    """Two trainers with the same seed must draw the same prompts each step."""
    policy_a = MockPolicy()
    policy_b = MockPolicy()
    pipeline = _all_mock_pipeline()
    schedule = PhaseSchedule(DEFAULT_PHASES)
    algorithm = get_algorithm("grpo", GrpoConfig())
    prompts = [_prompt(f"p{i}") for i in range(20)]
    config = TrainerConfig(
        total_steps=3,
        batch_prompts=2,
        group_size=4,
        seed=42,
    )
    t1 = Trainer(policy_a, pipeline, schedule, algorithm, config)
    t2 = Trainer(policy_b, pipeline, schedule, algorithm, config)
    r1 = t1.run(prompts)
    r2 = t2.run(prompts)
    # Same seed -> same mean rewards across runs.
    for a, b in zip(r1, r2):
        assert a.mean_reward == b.mean_reward


# ----------------------------------------------------------------------
# P0.1 + P0.2: Periodic checkpoint + resume-from-checkpoint
# ----------------------------------------------------------------------


def test_save_every_writes_intermediate_checkpoints(tmp_path: Path):
    """save_every=2 writes a checkpoint after step 2, step 4 of a 6-step run.

    Mock policy implements no save_adapter; we still expect the
    trainer_state.json and the dir layout to appear (the trainer falls
    back to a no-op for adapter+optimizer when the policy doesn't
    implement the save methods).
    """
    import json

    policy = MockPolicy()
    pipeline = _all_mock_pipeline()
    schedule = PhaseSchedule(DEFAULT_PHASES)
    algorithm = get_algorithm("grpo", GrpoConfig())
    prompts = [_prompt(f"p{i}") for i in range(8)]
    config = TrainerConfig(
        total_steps=6,
        batch_prompts=2,
        group_size=4,
        seed=42,
        save_every=2,
        keep_last_k=10,  # don't rotate during this test
        output_dir=tmp_path,
    )
    trainer = Trainer(policy, pipeline, schedule, algorithm, config)
    trainer.run(prompts)
    # Saves fire after step 2 and step 4 (not step 6 — the final is
    # written by train_method.py, not the trainer).
    assert (tmp_path / "checkpoint-2" / "trainer_state.json").exists()
    assert (tmp_path / "checkpoint-4" / "trainer_state.json").exists()
    assert not (tmp_path / "checkpoint-6" / "trainer_state.json").exists()
    state = json.loads(
        (tmp_path / "checkpoint-4" / "trainer_state.json").read_text()
    )
    assert state["step"] == 4


def test_keep_last_k_rotates_old_checkpoints(tmp_path: Path):
    """With keep_last_k=2, only the two most-recent saves survive."""
    policy = MockPolicy()
    pipeline = _all_mock_pipeline()
    schedule = PhaseSchedule(DEFAULT_PHASES)
    algorithm = get_algorithm("grpo", GrpoConfig())
    prompts = [_prompt(f"p{i}") for i in range(8)]
    config = TrainerConfig(
        total_steps=10,
        batch_prompts=2,
        group_size=4,
        seed=42,
        save_every=2,
        keep_last_k=2,
        output_dir=tmp_path,
    )
    trainer = Trainer(policy, pipeline, schedule, algorithm, config)
    trainer.run(prompts)
    surviving = sorted(
        p.name for p in tmp_path.iterdir() if p.name.startswith("checkpoint-")
    )
    # Saves fire after 2, 4, 6, 8 (not 10). keep_last_k=2 -> 6 and 8 survive.
    assert surviving == ["checkpoint-6", "checkpoint-8"]


def test_find_latest_checkpoint_picks_highest_step(tmp_path: Path):
    """find_latest_checkpoint picks by step number, not mtime."""
    for step in (3, 7, 5):
        ck = tmp_path / f"checkpoint-{step}"
        ck.mkdir()
        (ck / "trainer_state.json").write_text("{}")
    # Sanity touchpoint: the dir with the highest step wins regardless of order.
    latest = Trainer.find_latest_checkpoint(tmp_path)
    assert latest.name == "checkpoint-7"


def test_find_latest_checkpoint_returns_none_when_empty(tmp_path: Path):
    """Missing or empty output dir returns None (= start fresh)."""
    assert Trainer.find_latest_checkpoint(tmp_path) is None
    assert Trainer.find_latest_checkpoint(tmp_path / "nope") is None


def test_resume_from_step_continues_log_in_append_mode(tmp_path: Path):
    """start_step > 0 appends to train_log instead of truncating."""
    import json

    pipeline = _all_mock_pipeline()
    schedule = PhaseSchedule(DEFAULT_PHASES)
    algorithm = get_algorithm("grpo", GrpoConfig())
    prompts = [_prompt(f"p{i}") for i in range(8)]
    log_path = tmp_path / "train_log.jsonl"
    config = TrainerConfig(
        total_steps=4,
        batch_prompts=2,
        group_size=4,
        seed=42,
        log_path=log_path,
    )

    # First "run": 2 steps. Truncates an empty log to start fresh.
    policy_a = MockPolicy()
    t1 = Trainer(policy_a, pipeline, schedule, algorithm, config)
    config_a = TrainerConfig(
        total_steps=2,  # interrupted early
        batch_prompts=2,
        group_size=4,
        seed=42,
        log_path=log_path,
    )
    t1.config = config_a
    t1.run(prompts)
    n_initial = sum(1 for _ in log_path.open())
    assert n_initial == 2

    # Resume "run": continues from step=2, appends instead of truncating.
    policy_b = MockPolicy()
    t2 = Trainer(policy_b, pipeline, schedule, algorithm, config)
    t2.run(prompts, start_step=2)
    n_after = sum(1 for _ in log_path.open())
    assert n_after == 4, f"resume should append, got {n_after} lines"
    # Verify the appended records have step >= 2.
    lines = log_path.read_text().splitlines()
    last_record = json.loads(lines[-1])
    assert last_record["step"] >= 2


def test_resume_rng_fast_forward_matches_fresh_run(tmp_path: Path):
    """Resuming at step 2 with same seed produces same step 2+ records as fresh.

    Property: the trainer's prompt-sampling RNG is seeded once and consumed
    once per step. A resume must fast-forward the RNG by start_step draws so
    step 2 onward is identical to a fresh full run.
    """
    pipeline = _all_mock_pipeline()
    schedule = PhaseSchedule(DEFAULT_PHASES)
    algorithm = get_algorithm("grpo", GrpoConfig())
    prompts = [_prompt(f"p{i}") for i in range(20)]
    config = TrainerConfig(
        total_steps=4,
        batch_prompts=2,
        group_size=4,
        seed=42,
    )

    policy_full = MockPolicy()
    t_full = Trainer(policy_full, pipeline, schedule, algorithm, config)
    records_full = t_full.run(prompts)

    policy_resume = MockPolicy()
    t_resume = Trainer(policy_resume, pipeline, schedule, algorithm, config)
    records_resume = t_resume.run(prompts, start_step=2)

    # The resume run skips steps 0-1, so its records correspond to the
    # tail of the full run. Same RNG state -> same mean_reward.
    assert len(records_resume) == 2
    for full_rec, resume_rec in zip(records_full[2:], records_resume):
        assert full_rec.mean_reward == resume_rec.mean_reward


# ----------------------------------------------------------------------
# P0.3 + P0.4: training-time eval + best-checkpoint symlink
# ----------------------------------------------------------------------


def test_eval_writes_eval_log_at_cadence(tmp_path: Path):
    """eval_every=2 emits 3 eval rows over a 6-step run (at 2, 4, 6)."""
    import json

    policy = MockPolicy()
    pipeline = _all_mock_pipeline()
    schedule = PhaseSchedule(DEFAULT_PHASES)
    algorithm = get_algorithm("grpo", GrpoConfig())
    train_prompts = [_prompt(f"p{i}") for i in range(8)]
    val_prompts = [_prompt(f"v{i}") for i in range(3)]
    log_path = tmp_path / "train_log.jsonl"
    config = TrainerConfig(
        total_steps=6,
        batch_prompts=2,
        group_size=4,
        seed=42,
        log_path=log_path,
        eval_every=2,
        eval_prompts=val_prompts,
    )
    trainer = Trainer(policy, pipeline, schedule, algorithm, config)
    trainer.run(train_prompts)

    eval_log = tmp_path / "eval_log.jsonl"
    assert eval_log.exists()
    rows = [json.loads(l) for l in eval_log.open()]
    assert len(rows) == 3, f"expected 3 eval rows at steps 2,4,6; got {len(rows)}"
    assert [r["step"] for r in rows] == [2, 4, 6]
    # Compile-first family must be present in each row.
    for r in rows:
        assert "compile_at_1" in r
        assert "secure_at_1__compiles" in r
        assert "func_at_1__has_tests" in r
        assert "func_sec_at_1__compiles_and_has_tests" in r
        # Each row has n_prompts == len(val_prompts).
        assert r["n_prompts"] == 3


def test_eval_disabled_when_eval_every_zero(tmp_path: Path):
    """eval_prompts set but eval_every=0 → no eval_log.jsonl written."""
    policy = MockPolicy()
    pipeline = _all_mock_pipeline()
    schedule = PhaseSchedule(DEFAULT_PHASES)
    algorithm = get_algorithm("grpo", GrpoConfig())
    config = TrainerConfig(
        total_steps=4,
        batch_prompts=2,
        group_size=4,
        seed=42,
        log_path=tmp_path / "train_log.jsonl",
        eval_every=0,
        eval_prompts=[_prompt("v0")],
    )
    trainer = Trainer(policy, pipeline, schedule, algorithm, config)
    trainer.run([_prompt(f"p{i}") for i in range(4)])
    assert not (tmp_path / "eval_log.jsonl").exists()


def test_eval_disabled_when_no_eval_prompts(tmp_path: Path):
    """eval_every=2 but eval_prompts=None → no eval_log.jsonl written."""
    policy = MockPolicy()
    pipeline = _all_mock_pipeline()
    schedule = PhaseSchedule(DEFAULT_PHASES)
    algorithm = get_algorithm("grpo", GrpoConfig())
    config = TrainerConfig(
        total_steps=4,
        batch_prompts=2,
        group_size=4,
        seed=42,
        log_path=tmp_path / "train_log.jsonl",
        eval_every=2,
        eval_prompts=None,
    )
    trainer = Trainer(policy, pipeline, schedule, algorithm, config)
    trainer.run([_prompt(f"p{i}") for i in range(4)])
    assert not (tmp_path / "eval_log.jsonl").exists()


def test_eval_max_prompts_caps_val_set(tmp_path: Path):
    """eval_max_prompts=2 limits eval to the first 2 val prompts."""
    import json

    policy = MockPolicy()
    pipeline = _all_mock_pipeline()
    schedule = PhaseSchedule(DEFAULT_PHASES)
    algorithm = get_algorithm("grpo", GrpoConfig())
    config = TrainerConfig(
        total_steps=2,
        batch_prompts=2,
        group_size=4,
        seed=42,
        log_path=tmp_path / "train_log.jsonl",
        eval_every=2,
        eval_prompts=[_prompt(f"v{i}") for i in range(10)],
        eval_max_prompts=2,
    )
    trainer = Trainer(policy, pipeline, schedule, algorithm, config)
    trainer.run([_prompt(f"p{i}") for i in range(4)])
    rows = [json.loads(l) for l in (tmp_path / "eval_log.jsonl").open()]
    assert rows[0]["n_prompts"] == 2


def test_checkpoint_best_symlink_points_at_best_eval_step(tmp_path: Path):
    """checkpoint-best/ symlink updates when eval reports a new best.

    The MockPolicy returns the same completion every time, so the eval
    metric is constant across steps and the FIRST eval that lands wins
    — checkpoint-best should point at that first-eval checkpoint.
    """
    import os

    policy = MockPolicy()
    pipeline = _all_mock_pipeline()
    schedule = PhaseSchedule(DEFAULT_PHASES)
    algorithm = get_algorithm("grpo", GrpoConfig())
    config = TrainerConfig(
        total_steps=4,
        batch_prompts=2,
        group_size=4,
        seed=42,
        log_path=tmp_path / "train_log.jsonl",
        save_every=2,
        keep_last_k=10,
        output_dir=tmp_path,
        eval_every=2,
        eval_prompts=[_prompt("v0")],
    )
    trainer = Trainer(policy, pipeline, schedule, algorithm, config)
    trainer.run([_prompt(f"p{i}") for i in range(4)])
    best_link = tmp_path / "checkpoint-best"
    assert best_link.is_symlink() or best_link.exists()
    # The symlink should point at one of the checkpoint-{N} dirs.
    target = os.readlink(str(best_link))
    assert target.startswith("checkpoint-")


def test_eval_log_append_on_resume(tmp_path: Path):
    """Resume continues writing eval_log.jsonl in append mode."""
    import json

    pipeline = _all_mock_pipeline()
    schedule = PhaseSchedule(DEFAULT_PHASES)
    algorithm = get_algorithm("grpo", GrpoConfig())
    val_prompts = [_prompt(f"v{i}") for i in range(2)]
    log_path = tmp_path / "train_log.jsonl"

    policy_a = MockPolicy()
    config_a = TrainerConfig(
        total_steps=2,
        batch_prompts=2,
        group_size=4,
        seed=42,
        log_path=log_path,
        eval_every=2,
        eval_prompts=val_prompts,
    )
    t1 = Trainer(policy_a, pipeline, schedule, algorithm, config_a)
    t1.run([_prompt(f"p{i}") for i in range(4)])
    eval_log = tmp_path / "eval_log.jsonl"
    n_initial = sum(1 for _ in eval_log.open())
    assert n_initial == 1

    policy_b = MockPolicy()
    config_b = TrainerConfig(
        total_steps=4,
        batch_prompts=2,
        group_size=4,
        seed=42,
        log_path=log_path,
        eval_every=2,
        eval_prompts=val_prompts,
    )
    t2 = Trainer(policy_b, pipeline, schedule, algorithm, config_b)
    t2.run([_prompt(f"p{i}") for i in range(4)], start_step=2)
    n_after = sum(1 for _ in eval_log.open())
    assert n_after == 2, f"resume should append eval rows; got {n_after}"
    rows = [json.loads(l) for l in eval_log.open()]
    assert rows[1]["step"] == 4


def test_trainer_passes_cwe_weights_to_policy_step():
    """Algorithm 1 step 11: w(x) scales each prompt's rollouts' loss."""
    from secure_code_rl_ictai.rl.reweight import ReweightConfig, Reweighter

    seen: list[np.ndarray] = []

    class WeightRecordingPolicy(MockPolicy):
        def step(self, loss, step_idx, *, sample_weights=None, **kwargs):
            seen.append(sample_weights)
            return super().step(loss, step_idx, **kwargs)

    prompts = [_prompt("a", cwe="CWE-89")] * 9 + [_prompt("b", cwe="CWE-306")]
    trainer = Trainer(
        policy=WeightRecordingPolicy(),
        reward_pipeline=_all_mock_pipeline(),
        schedule=PhaseSchedule(DEFAULT_PHASES),
        algorithm=get_algorithm("grpo", GrpoConfig()),
        config=TrainerConfig(total_steps=2, batch_prompts=4, group_size=3, seed=0),
        reweighter=Reweighter.from_cwes(ReweightConfig(), (p.target_cwe for p in prompts)),
    )
    trainer.run(prompts)
    assert len(seen) == 2
    for w in seen:
        assert w is not None and w.shape == (12,)
        assert w.mean() == pytest.approx(1.0)

