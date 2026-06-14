#!/bin/bash
# 4 x 2 algorithm x retrieval factorial (paper Table IV) = 8 training cells.
#
# Every cell uses train_method.py's defaults, which are the paper's main
# configuration (Section V-D), and differs only in algorithm and R_RAG:
#   arm_a_grpo / arm_a_grpo_rag   GRPO, SAST only / + R_RAG
#   arm_a_ppo  / arm_a_ppo_rag    value-baselined PPO (G = 8)
#   arm_a_rloo / arm_a_rloo_rag   RLOO
#   arm_a_raft / arm_a_raft_rag   RAFT
# The SAST-only condition keeps the reliability reward, stub penalty, CWE
# reweighting, and sigma floor, and removes only R_RAG.

set -euo pipefail

REPO=/storage/home/sss6371/secure-code-rl-ictai
SCRATCH_ROOT=/scratch/sss6371/secure-code-rl-ictai-data
TRAIN_JSONL=$SCRATCH_ROOT/build/v0.1.7/train_prompts.jsonl
VAL_JSONL=$SCRATCH_ROOT/build/v0.1.7/val_prompts.jsonl
RAG_INDEX_DIR=$SCRATCH_ROOT/build/v0.1.7/rag_index
OUT_ROOT=$SCRATCH_ROOT/v0_1_7
LOGS=$REPO/logs


if [[ ! -d "$RAG_INDEX_DIR" ]]; then
    echo "ERROR: $RAG_INDEX_DIR not found." >&2
    exit 1
fi

mkdir -p "$OUT_ROOT" "$LOGS"

# Shared trainer hyperparameters. Per FINDINGS_LOG 2026-06-14, the value
# head adds ~3GB of memory on top of the existing LoRA+ref+optimizer
# footprint. batch=1 stays since the OOM ceiling on a100-40GB hasn't
# moved.
COMMON_FLAGS=(
    --train-jsonl "$TRAIN_JSONL"
    --total-steps 1000
    --group-size 16
    --max-new-tokens 512
    --temperature 0.7
    --seed 42
    --lora-r 16
    --lora-alpha 32
    --save-every 200
    --keep-last-k 3
    --eval-jsonl "$VAL_JSONL"
    --eval-every 100
    --eval-max-prompts 0
)

# FIRE gate: this script fires unconditionally and was accidentally
# re-submitting every dry-run inspection. Require explicit FIRE=1.
FIRE=${FIRE:-0}

# Each cell needs a hf_cache dir per FINDINGS_LOG (concurrent HF fetches
# clobber each other otherwise).
N_CHAIN=${N_CHAIN:-1}

# Resume strategy: only chain-index 1 cold-starts. The rest auto-resume
# from the previous job's last checkpoint. This avoids both silent
# cold-restart AND stale-state resume bugs.
submit_chain() {
    local name=$1
    local extra=$2
    local gpu=${3:-a100}  # PPO uses a40 (45GB) to fit value head at 512 tokens
    local out=$OUT_ROOT/$name
    local hf=$SCRATCH_ROOT/hf_cache/$name
    if [[ "$FIRE" != "1" ]]; then
        echo "[DRY] would submit $N_CHAIN-job chain: $name (output=$out, gpu=$gpu)"
        return 0
    fi
    mkdir -p "$out" "$hf"

    local dep=""
    for i in $(seq 1 "$N_CHAIN"); do
        local resume_flag=""
        if [[ "$i" -ne 1 ]]; then
            resume_flag="--resume-from $out"
        fi
        local jid
        jid=$(sbatch --parsable $dep \
            --account=szs339_cr_default \
            --partition=standard \
            --gres=gpu:${gpu}:1 \
            --time=4320:00 \
            --mem=80G \
            --output="$LOGS/${name}_%j.log" \
            --job-name="ictai_$name" \
            --wrap="set -a; [ -f $REPO/.env ] && . $REPO/.env; set +a; export HF_HOME=$hf && export PATH=/storage/home/sss6371/.local/share/codeql:$REPO/.venv/bin:/storage/work/sss6371/.conda/envs/sast/bin:\$PATH && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && cd $REPO && PYTHONPATH=src .venv/bin/python scripts/train_method.py ${COMMON_FLAGS[*]} --output $out $extra $resume_flag")
        echo "  $name: job $jid (i=$i/$N_CHAIN, gpu=$gpu)"
        dep="--dependency=afterany:$jid"
    done
}

RAG_FLAGS="--reward-rag-on --rag-index-dir $RAG_INDEX_DIR --lambda-rag 0.1"

submit_chain "arm_a_grpo"     "--algorithm grpo"
submit_chain "arm_a_grpo_rag" "--algorithm grpo $RAG_FLAGS"

# PPO carries a value head, so it runs at G = 8 on a40 (45 GB).
submit_chain "arm_a_ppo"      "--algorithm ppo --group-size 8" a40
submit_chain "arm_a_ppo_rag"  "--algorithm ppo --group-size 8 $RAG_FLAGS" a40

submit_chain "arm_a_rloo"     "--algorithm rloo"
submit_chain "arm_a_rloo_rag" "--algorithm rloo $RAG_FLAGS"

submit_chain "arm_a_raft"     "--algorithm raft"
submit_chain "arm_a_raft_rag" "--algorithm raft $RAG_FLAGS"

echo
echo "Submitted 4 × 2 factorial = 8 cells × ${N_CHAIN} chain jobs each."
echo "Total queued: $((8 * N_CHAIN)) SLURM jobs (long afterany chains)."
echo
echo "Watch with:"
echo "  squeue --user=\$USER --name=ictai_arm_a_* --format='%.10i %.25j %.2t %.10M'"
echo "  bash scripts/monitor_v0_1_7_factorial.sh   # if you wire one up"
echo
echo "Outputs under: $OUT_ROOT/{grpo,grpo_rag,ppo,ppo_rag,rloo,rloo_rag,raft,raft_rag}/"
echo
echo "Eval lands at \$out/eval_log.jsonl after every 100 steps."
echo "Check policy_loss_nonzero rolling rate in train_log.jsonl to detect"
echo "dead cells early (FINDINGS_LOG 2026-06-14 silent-collapse fix)."
