#!/bin/bash
# Table III "+ SFT warm start": CARGO (GRPO + R_RAG) initialized from the
# lowest-validation-loss adapter of a 20000-step SFT run on the training
# pool (scripts/train_sft.py).

set -euo pipefail

REPO=/storage/home/sss6371/secure-code-rl-ictai
SCRATCH_ROOT=/scratch/sss6371/secure-code-rl-ictai-data
TRAIN_JSONL=$SCRATCH_ROOT/build/v0.1.7/train_prompts.jsonl
VAL_JSONL=$SCRATCH_ROOT/build/v0.1.7/val_prompts.jsonl
RAG_INDEX_DIR=$SCRATCH_ROOT/build/v0.1.7/rag_index
OUT_ROOT=$SCRATCH_ROOT/v0_1_7_sft_warm_start
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
    local overrides=$2
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
            --wrap="set -a; [ -f $REPO/.env ] && . $REPO/.env; set +a; export HF_HOME=$hf && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && cd $REPO && PYTHONPATH=src .venv/bin/python scripts/train_method.py ${COMMON_FLAGS[*]} --output $out $overrides $([[ "$overrides" == *--warm-start-adapter* ]] && echo "" || echo "--resume-from $out")")
        echo "  $name: job $jid (i=$i/$N_CHAIN)"
        dep="--dependency=afterany:$jid"
    done
}

# C2: SFT warm start (Table III). Run scripts/train_sft.py for 20000 steps
# on the training pool first; the lowest-validation-loss checkpoint is
# checkpoint-best. Runs at G = 8 for memory.
SFT_CKPT=${SFT_CKPT:-$SCRATCH_ROOT/runs/sft_warmstart_v0_1_7/checkpoint-best}
if [[ -d "$SFT_CKPT/adapter" ]]; then
    submit_chain "tierC2_sft_warm" \
        "--reward-rag-on --rag-index-dir $RAG_INDEX_DIR --lambda-rag 0.1 --group-size 8 --warm-start-adapter $SFT_CKPT"
else
    echo "SKIP C2 (SFT warm-start): $SFT_CKPT/adapter not found."
    echo "  Run scripts/train_sft.py --total-steps 20000 first."
fi
