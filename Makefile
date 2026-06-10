# CARGO (ICTAI 2026) — secure code generation via RL with a retrieval-grounded reward.
#
# One-invocation targets for local dev and ROAR validation. Most contributors
# only need `make test` (locally) and `make test-roar` (after a sync).

# --- knobs ---

PYTHON ?= python3
PYTEST ?= pytest

ROAR_HOST ?= roar
ROAR_PATH ?= /storage/home/sss6371/secure-code-rl-ictai
ROAR_VENV_PYTEST ?= .venv/bin/pytest

# rsync excludes: don't push local caches, git history, or large data.
# Keep small static artifacts (data/.gitkeep, data/nvdlib_cwe_medians.json).
RSYNC_EXCLUDES = \
	--exclude='.git' \
	--exclude='__pycache__' \
	--exclude='.pytest_cache' \
	--exclude='.ruff_cache' \
	--exclude='.mypy_cache' \
	--exclude='*.pyc' \
	--exclude='.venv' \
	--exclude='.venv_*' \
	--exclude='.env' \
	--exclude='.env.*' \
	--exclude='wandb' \
	--exclude='runs' \
	--exclude='results' \
	--exclude='logs'

.PHONY: help
help:
	@echo "ICTAI Makefile targets:"
	@echo "  test            Run the fast unit-test suite locally (default markers)."
	@echo "  test-real       Run unit + real-oracle tests locally."
	@echo "  test-all        Run every test locally (incl. real_sast; needs SAST tools)."
	@echo "  sync-roar       rsync the repo to \$$ROAR_HOST:\$$ROAR_PATH (excludes .git, caches, data)."
	@echo "  test-roar       Sync then run the fast unit-test suite on ROAR."
	@echo "  test-roar-real  Sync then run real-oracle tests on ROAR (Python-only; C tests skip)."
	@echo "  roar-status     Print SLURM queue + disk + raw data sizes on ROAR."
	@echo "  roar-build-rag-index Sync, then submit build_rag_index.py on ROAR with a40 GPU."
	@echo "  lint            Run ruff (if installed)."
	@echo "  format          Run ruff format (if installed)."
	@echo "  clean           Remove caches and build artifacts."
	@echo ""
	@echo "Overrides:"
	@echo "  ROAR_HOST=roar (default)   ROAR_PATH=$(ROAR_PATH)"
	@echo "  ROAR_DATA_ROOT=$(ROAR_DATA_ROOT)   ROAR_BUILD_VERSION=$(ROAR_BUILD_VERSION)"

# --- local ---

.PHONY: test
test:
	PYTHONPATH=src $(PYTEST) tests/ -v

.PHONY: test-real
test-real:
	PYTHONPATH=src $(PYTEST) tests/ -v -m "real_oracle or not (real_oracle or real_sast)"

.PHONY: test-all
test-all:
	PYTHONPATH=src $(PYTEST) tests/ -v -m "real_oracle or real_sast or not (real_oracle or real_sast)"

.PHONY: lint
lint:
	@command -v ruff >/dev/null 2>&1 \
		&& ruff check src tests scripts \
		|| echo "ruff not installed; skipping"

.PHONY: format
format:
	@command -v ruff >/dev/null 2>&1 \
		&& ruff format src tests scripts \
		|| echo "ruff not installed; skipping"

.PHONY: clean
clean:
	find . -type d -name '__pycache__' -prune -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name '.pytest_cache' -prune -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name '.ruff_cache' -prune -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name '.mypy_cache' -prune -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.pyc' -delete 2>/dev/null || true

# --- ROAR ---

.PHONY: sync-roar
sync-roar:
	@echo ">>> rsync to $(ROAR_HOST):$(ROAR_PATH)/"
	rsync -av --delete $(RSYNC_EXCLUDES) ./ $(ROAR_HOST):$(ROAR_PATH)/
	@echo ">>> sync complete"

.PHONY: test-roar
test-roar: sync-roar
	@echo ">>> running fast unit-test suite on $(ROAR_HOST)"
	ssh $(ROAR_HOST) 'cd $(ROAR_PATH) && PYTHONPATH=src $(ROAR_VENV_PYTEST) tests/ -v'

.PHONY: test-roar-real
test-roar-real: sync-roar
	@echo ">>> running real-oracle tests on $(ROAR_HOST)"
	ssh $(ROAR_HOST) 'cd $(ROAR_PATH) && PYTHONPATH=src $(ROAR_VENV_PYTEST) tests/ -v -m "real_oracle or not (real_oracle or real_sast)"'

# --- ROAR dataset build orchestration ---
#
# Variables for the build-dataset workflow. Override per session.
ROAR_DATA_ROOT ?= /storage/home/sss6371/work/secure-code-rl-ictai-data/raw
ROAR_BUILD_VERSION ?= v0.1.7

# GPU choice. Per measured benchmarks (2026-06-10) and project policy
# (memory: ictai-gpu-choice):
#   - ROAR_GPU_TRAIN: batched training (METHOD baselines, GRPO/PPO with
#     K>1 rollouts). a100 wins here by 2-3x — tensor cores need batched
#     matmuls to amortize call overhead.
#   - ROAR_GPU_EVAL:  greedy single-prompt inference (MULTI_LLM sweep on
#     1411 eval prompts, ablation eval, calibration). a40 wins here by
#     ~30% over a100 because batch=1 is memory-bandwidth bound and a40
#     GDDR6 + SM86 actually beats a100 HBM2e at small shapes. Also
#     consistently available (12 a40 nodes; ~instant scheduling).
ROAR_GPU_TRAIN ?= a100:1
ROAR_GPU_EVAL  ?= a40:1
# legacy aliases (do not use in new targets)
ROAR_GPU_REAL ?= $(ROAR_GPU_TRAIN)
ROAR_GPU_TEST ?= $(ROAR_GPU_EVAL)

.PHONY: roar-status
roar-status:
	@echo ">>> queue + storage on $(ROAR_HOST)"
	ssh $(ROAR_HOST) 'squeue --user=sss6371; echo "---disk---"; df -h /storage/home/sss6371/work/ | head -2; echo "---data sizes---"; du -sh $(ROAR_DATA_ROOT)/*'



.PHONY: roar-build-rag-index
roar-build-rag-index: sync-roar
	@echo ">>> submitting build_rag_index on $(ROAR_HOST) (a40 GPU; one-off ~4 min, no need for a100)"
	ssh $(ROAR_HOST) 'sbatch --wrap="cd $(ROAR_PATH) && PYTHONPATH=src .venv/bin/python scripts/build_rag_index.py \
		--exemplar-pairs $(ROAR_DATA_ROOT)/../build/$(ROAR_BUILD_VERSION)/exemplar_pairs.jsonl \
		--output $(ROAR_DATA_ROOT)/../build/$(ROAR_BUILD_VERSION)/rag_index \
		--embedder hf --device auto --force" \
		--partition=standard --gres=gpu:$(ROAR_GPU_TEST) --time=120:00 --mem=64G \
		--output=$(ROAR_PATH)/logs/build_rag_index_%j.log \
		--job-name=ictai_build_rag_index'

# --- METHOD training ------------------------------------------------------
# Smoke: 5 steps on a40 to validate plumbing (no LoRA weights persisted in
# the smoke run by design — we only confirm the loop runs end-to-end).
# Full: one cell with train_method.py's defaults (the paper's main configuration).

.PHONY: roar-train-smoke
roar-train-smoke: sync-roar
	@echo ">>> submitting train smoke on $(ROAR_HOST) (a40, 5 steps)"
	ssh $(ROAR_HOST) 'sbatch --wrap="cd $(ROAR_PATH) && PYTHONPATH=src .venv/bin/python scripts/train_method.py \
		--train-jsonl $(ROAR_DATA_ROOT)/../build/$(ROAR_BUILD_VERSION)/train_prompts.jsonl \
		--max-prompts 50 \
		--output $(ROAR_DATA_ROOT)/../runs/smoke_grpo_5 \
		--total-steps 5 --batch-prompts 1 --group-size 2 --max-new-tokens 256 \
		--algorithm grpo" \
		--partition=standard --gres=gpu:$(ROAR_GPU_TEST) --time=60:00 --mem=64G \
		--output=$(ROAR_PATH)/logs/train_smoke_%j.log \
		--job-name=ictai_train_smoke'

.PHONY: roar-train-method
roar-train-method: sync-roar
	@echo ">>> submitting full METHOD cell on $(ROAR_HOST) (a100, 1000 steps)"
	@echo "    cell: ROAR_METHOD_CELL=$(ROAR_METHOD_CELL); algorithm=$(ROAR_ALGORITHM)"
	ssh $(ROAR_HOST) 'sbatch --wrap="cd $(ROAR_PATH) && PYTHONPATH=src .venv/bin/python scripts/train_method.py \
		--train-jsonl $(ROAR_DATA_ROOT)/../build/$(ROAR_BUILD_VERSION)/train_prompts.jsonl \
		--output $(ROAR_DATA_ROOT)/../runs/$(ROAR_METHOD_CELL) \
		--algorithm $(ROAR_ALGORITHM)" \
		--partition=standard --gres=gpu:$(ROAR_GPU_TRAIN) --time=2880:00 --mem=128G \
		--output=$(ROAR_PATH)/logs/train_$(ROAR_METHOD_CELL)_%j.log \
		--job-name=ictai_train_$(ROAR_METHOD_CELL)'

# Defaults for the training targets.
ROAR_METHOD_CELL ?= v0.1.7_grpo
ROAR_ALGORITHM   ?= grpo
