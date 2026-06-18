#!/bin/bash
# Paper controls and replicates on top of CARGO (GRPO + R_RAG), one
# sbatch job per cell. Every cell inherits train_method.py's paper
# defaults plus the retrieval reward and overrides one knob:
#   seed_1337, seed_2024                         seed replicates (Section VI-A)
#   sigma_floor_{0,0p01,0p1}, sigma_floor_anneal  sigma_min sweep (Section VI-C)
#   rag_adversarial, tierC1_random_rag,
#   rag_binary, rag_binary_random                retrieval-reward 2x2 and controls
#   tierE1_qwen3b, tierE2_starcoder3b            cross-family reproduction (Table V)
#
# Usage:
#   bash scripts/queue_post_headline_bundle.sh           # dry-run (print + skip)
#   FIRE=1 bash scripts/queue_post_headline_bundle.sh    # actually submit
#
# Idempotent: skips cells whose --output dir already has a train log or a
# job in the queue.

set -euo pipefail

REPO=/storage/home/sss6371/secure-code-rl-ictai
SCRATCH_ROOT=/scratch/sss6371/secure-code-rl-ictai-data
TRAIN_JSONL=$SCRATCH_ROOT/build/v0.1.7/train_prompts.jsonl
VAL_JSONL=$SCRATCH_ROOT/build/v0.1.7/val_prompts.jsonl
RAG_INDEX_DIR=$SCRATCH_ROOT/build/v0.1.7/rag_index
RAG_INDEX_MIN_DIFF_DIR=$SCRATCH_ROOT/build/v0.1.7/rag_index_min_diff
OUT_ROOT=$SCRATCH_ROOT/v0_1_7_bundle
LOGS=$REPO/logs


mkdir -p "$OUT_ROOT" "$LOGS"

# Default flags shared across cells. Replicates the headline recipe
# unless a per-cell override changes one specific argument.
DEFAULT_FLAGS=(
    --train-jsonl "$TRAIN_JSONL"
    --algorithm grpo
    --total-steps 1000
    --group-size 16
    --max-new-tokens 512
    --temperature 0.7
    --lora-r 16
    --lora-alpha 32
    --save-every 200
    --keep-last-k 3
    --eval-jsonl "$VAL_JSONL"
    --eval-every 100
    --eval-max-prompts 0
    --reward-rag-on
    --rag-index-dir "$RAG_INDEX_DIR"
    --lambda-rag 0.1
)

FIRE=${FIRE:-0}

# submit <name> <override_flags>
# Override flags are appended; argparse last-value-wins makes them stick.
submit() {
    local name=$1
    local overrides=$2
    local out=$OUT_ROOT/$name
    local hf=$SCRATCH_ROOT/hf_cache/$name

    # Idempotency: skip if any of these hold:
    #   - a checkpoint-N dir exists (cell has progressed)
    #   - train_log.jsonl exists (cell is mid-stream, may not have hit
    #     checkpoint-100 yet)
    #   - there's a job in squeue for this cell name (avoid racing)
    if compgen -G "$out/checkpoint-[1-9]*" > /dev/null 2>&1; then
        echo "SKIP $name: checkpoint already exists in $out"
        return 0
    fi
    if [[ -f "$out/train_log.jsonl" ]]; then
        echo "SKIP $name: train_log.jsonl already exists in $out (cell mid-stream)"
        return 0
    fi
    if squeue --user="$USER" --noheader --format="%j" 2>/dev/null \
            | grep -q "^ictai_bundle_$name$"; then
        echo "SKIP $name: job already in squeue"
        return 0
    fi

    if [[ "$FIRE" != "1" ]]; then
        echo "[DRY] would submit: $name"
        echo "[DRY]   overrides: $overrides"
        return 0
    fi

    mkdir -p "$out" "$hf"
    local jid
    jid=$(sbatch --parsable \
        --account=szs339_cr_default \
        --partition=standard \
        --gres=gpu:a100:1 \
        --time=4320:00 \
        --mem=80G \
        --output="$LOGS/bundle_${name}_%j.log" \
        --job-name="ictai_bundle_$name" \
        --wrap="set -a; [ -f $REPO/.env ] && . $REPO/.env; set +a; export HF_HOME=$hf && export PATH=/storage/home/sss6371/.local/share/codeql:$REPO/.venv/bin:/storage/work/sss6371/.conda/envs/sast/bin:\$PATH && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && cd $REPO && PYTHONPATH=src .venv/bin/python scripts/train_method.py ${DEFAULT_FLAGS[*]} --output $out $overrides")
    echo "FIRED $name: job $jid -> $out"
}

submit "seed_1337" "--seed 1337"
submit "seed_2024" "--seed 2024"

# sigma_min sweep around the main 0.05 (Table III "- sigma-floor" is 0).
submit "sigma_floor_0" "--seed 42 --sigma-floor 0.0"
submit "sigma_floor_0p01" "--seed 42 --sigma-floor 0.01"
submit "sigma_floor_0p1" "--seed 42 --sigma-floor 0.1"
submit "sigma_floor_anneal" \
    "--seed 42 --sigma-floor-initial 0.1 --sigma-floor-final 0.01 --sigma-floor-anneal-steps 1000"

# ----- Tier E1 Qwen2.5-Coder-3B reproduction -----
# Bigger model, longer per-step wall. 3-day walltime accommodates.
# tierE1 (3B model) needs G=8 instead of G=16 on a100-40GB: at G=16
# the 3B base + LoRA + ref + KV cache + activations exceeds 40GB.
# Keep max-new-tokens=512 (inherited from DEFAULT_FLAGS) for token-budget
# consistency with the 1.5B cells; only the group-size differs.
submit "tierE1_qwen3b" \
    "--seed 42 --model-id Qwen/Qwen2.5-Coder-3B-Instruct --group-size 8"

# ----- Tier E2 StarCoder2-3B reproduction -----
# Different model family. Tests whether the recipe generalizes off
# the Qwen2.5 line. Same G=8 reduction as tierE1 to fit a100-40GB
# (3B base + LoRA + ref + KV cache at G=16 + 512 tokens > 40GB).
submit "tierE2_starcoder3b" \
    "--seed 42 --model-id bigcode/starcoder2-3b --group-size 8"

# ----- #269 adversarial-retrieval RAG -----
# Worst-match retriever. Tests "does retrieval quality matter"
# (continuous-signal-density vs retrieval-quality decomposition).
submit "rag_adversarial" \
    "--seed 42 --rag-retriever-mode adversarial"

# ----- Random retrieval: exemplar sampled uniformly from any CWE -----
submit "tierC1_random_rag" \
    "--seed 42 --rag-retriever-mode random"

# ----- Binary control: 1[cos(y, e+) > cos(y, e-)], CWE-matched and random -----
submit "rag_binary" \
    "--seed 42 --rag-binary"
submit "rag_binary_random" \
    "--seed 42 --rag-binary --rag-retriever-mode random"


echo
if [[ "$FIRE" == "1" ]]; then
    echo "Submitted bundle. Watch:"
    echo "  watch squeue --user=\$USER --format='%.10i %.30j %.2t %.10M'"
    echo "  cd $REPO && python3 scripts/monitor_factorial_cells.py  (after editing path)"
else
    echo "DRY RUN ONLY. Set FIRE=1 to actually submit:"
    echo "  FIRE=1 bash scripts/queue_post_headline_bundle.sh"
fi
