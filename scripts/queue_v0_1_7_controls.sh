#!/bin/bash
# Table III "- CWE reweight": CARGO (GRPO + R_RAG) with uniform per-prompt
# weights. Pairs with arm_a_grpo_rag from queue_v0_1_7_factorial.sh.

set -euo pipefail

REPO=/storage/home/sss6371/secure-code-rl-ictai
SCRATCH_ROOT=/scratch/sss6371/secure-code-rl-ictai-data
TRAIN_JSONL=$SCRATCH_ROOT/build/v0.1.7/train_prompts.jsonl
VAL_JSONL=$SCRATCH_ROOT/build/v0.1.7/val_prompts.jsonl
RAG_INDEX_DIR=$SCRATCH_ROOT/build/v0.1.7/rag_index
OUT_ROOT=$SCRATCH_ROOT/v0_1_7
LOGS=$REPO/logs


mkdir -p "$OUT_ROOT" "$LOGS"

COMMON_FLAGS=(
    --train-jsonl "$TRAIN_JSONL"
    --algorithm grpo
    --total-steps 1000
    --group-size 16
    --max-new-tokens 512
    --temperature 0.7
    --seed 42
    --lora-r 16
    --lora-alpha 32
    --save-every 100
    --keep-last-k 5
    --eval-jsonl "$VAL_JSONL"
    --eval-every 100
    --eval-max-prompts 0
)

N_CHAIN=${N_CHAIN:-1}

submit_chain() {
    local name=$1
    local extra=$2
    local out=$OUT_ROOT/$name
    local hf=$SCRATCH_ROOT/hf_cache/$name
    mkdir -p "$out" "$hf"

    local dep=""
    for i in $(seq 1 "$N_CHAIN"); do
        local jid
        jid=$(sbatch --parsable $dep \
            --account=szs339_cr_default \
            --partition=standard \
            --gres=gpu:a100:1 \
            --time=4320:00 \
            --mem=80G \
            --output="$LOGS/${name}_%j.log" \
            --job-name="ictai_$name" \
            --wrap="set -a; [ -f $REPO/.env ] && . $REPO/.env; set +a; export HF_HOME=$hf && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && cd $REPO && PYTHONPATH=src .venv/bin/python scripts/train_method.py ${COMMON_FLAGS[*]} --output $out $extra --resume-from $out")
        echo "  $name: job $jid (i=$i/$N_CHAIN)"
        dep="--dependency=afterany:$jid"
    done
}

# Uniform-RAG: reweight off, continuous R_RAG on. Pairs with arm_a_grpo_rag
# (reweight on, continuous R_RAG on) for the §7.2 contrast.
submit_chain "arm_a_grpo_uniform_rag" \
    "--reweight-enabled false --reward-rag-on --rag-index-dir $RAG_INDEX_DIR --lambda-rag 0.1"

echo
echo "Submitted 1 control cell × ${N_CHAIN} chain jobs."
echo
echo "Output under: $OUT_ROOT/grpo_uniform_rag/"
