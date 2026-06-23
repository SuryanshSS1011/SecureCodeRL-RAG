#!/bin/bash
# RL headline-eval watcher.
# Polls 22 RL cells for checkpoint-best (or checkpoint-1000), fires
# scripts/headline_eval.py per cell as each finishes. De-dups via STATE
# array so each cell fires exactly once.
#
# Output: per-cell aggregate.json under
#   /scratch/sss6371/secure-code-rl-ictai-data/sweeps/phase1_v0_1_7_headline/<cell>/

set -u
LOG=/storage/home/sss6371/secure-code-rl-ictai/logs/rl_headline_eval_watcher.log
INSTANCE_LOCK=/storage/home/sss6371/secure-code-rl-ictai/logs/rl_headline_eval_watcher.instance.lock
FIRE_LOCK=/storage/home/sss6371/secure-code-rl-ictai/logs/rl_headline_eval_watcher.fire.lock
exec >> "$LOG" 2>&1

# Cross-node single-instance guard. flock on a shared NFS file is the
# only reliable mutex across multiple submit hosts. Previous PID-file
# guard only worked within one node.
exec 9>"$INSTANCE_LOCK"
if ! flock -n 9; then
    echo "[$(date)] another watcher already holds $INSTANCE_LOCK on some node; exiting"
    exit 0
fi
echo "[$(date)] watcher start, pid=$$ host=$(hostname)"

REPO=/storage/home/sss6371/secure-code-rl-ictai
SCRATCH=/scratch/sss6371/secure-code-rl-ictai-data
EVAL_JSONL=$SCRATCH/build/v0.1.7/eval_prompts.jsonl
RAG_INDEX_DIR=$SCRATCH/build/v0.1.7/rag_index
OUT_ROOT=$SCRATCH/sweeps/phase1_v0_1_7_headline
HF_CACHE=$SCRATCH/hf_cache/headline_eval_v0_1_7
mkdir -p "$OUT_ROOT" "$HF_CACHE"

# All 22 cells: name -> output-dir-of-training-run
declare -A CELLS=(
    [arm_a_grpo]=$SCRATCH/v0_1_7/arm_a_grpo
    [arm_a_grpo_rag]=$SCRATCH/v0_1_7/arm_a_grpo_rag
    [arm_a_ppo]=$SCRATCH/v0_1_7/arm_a_ppo
    [arm_a_ppo_rag]=$SCRATCH/v0_1_7/arm_a_ppo_rag
    [arm_a_rloo]=$SCRATCH/v0_1_7/arm_a_rloo
    [arm_a_rloo_rag]=$SCRATCH/v0_1_7/arm_a_rloo_rag
    [arm_a_raft]=$SCRATCH/v0_1_7/arm_a_raft
    [arm_a_raft_rag]=$SCRATCH/v0_1_7/arm_a_raft_rag
    [bundle_seed_1337]=$SCRATCH/v0_1_7_bundle/seed_1337
    [bundle_seed_2024]=$SCRATCH/v0_1_7_bundle/seed_2024
    [bundle_sigma_floor_0]=$SCRATCH/v0_1_7_bundle/sigma_floor_0
    [bundle_tierE1_qwen3b]=$SCRATCH/v0_1_7_bundle/tierE1_qwen3b
    [bundle_tierE2_starcoder3b]=$SCRATCH/v0_1_7_bundle/tierE2_starcoder3b
    [bundle_rag_adversarial]=$SCRATCH/v0_1_7_bundle/rag_adversarial
    [bundle_tierC1_random_rag]=$SCRATCH/v0_1_7_bundle/tierC1_random_rag
    [bundle_rag_binary]=$SCRATCH/v0_1_7_bundle/rag_binary
    [bundle_sigma_floor_0p01]=$SCRATCH/v0_1_7_bundle/sigma_floor_0p01
    [bundle_sigma_floor_0p1]=$SCRATCH/v0_1_7_bundle/sigma_floor_0p1
    [bundle_sigma_floor_anneal]=$SCRATCH/v0_1_7_bundle/sigma_floor_anneal
    [bundle_rag_binary_random]=$SCRATCH/v0_1_7_bundle/rag_binary_random
    [tierC2_sft_warm]=$SCRATCH/v0_1_7_sft_warm_start/tierC2_sft_warm
    # SFT-only cells (added 2026-06-22 after chat-template fix retrain).
    # Each cell trains 20000 steps, writes checkpoint-best symlink under
    # the cell dir.
    [sft_r16]=$SCRATCH/v0_1_7_sft_only_r16
    [sft_r32]=$SCRATCH/v0_1_7_sft_only_r32
    [sft_r64]=$SCRATCH/v0_1_7_sft_only_r64
    [sft_r128]=$SCRATCH/v0_1_7_sft_only_r128
    [sft_r256]=$SCRATCH/v0_1_7_sft_only_r256
    [sft_r512]=$SCRATCH/v0_1_7_sft_only_r512
    [sft_fullft]=$SCRATCH/v0_1_7_sft_only_fullft
    [arm_a_grpo_uniform_rag]=$SCRATCH/v0_1_7/arm_a_grpo_uniform_rag
)

declare -A STATE=()
# Pre-detect already-fired jobs so a restart doesn't re-fire.
for name in "${!CELLS[@]}"; do
    if squeue -u $USER -o "%j" --noheader 2>/dev/null | grep -qx "ictai_he_v017_${name}"; then
        STATE[$name]=fired
        echo "[$(date)] $name: eval job already in queue, marking fired"
    elif [ -f "$OUT_ROOT/$name/aggregate.json" ]; then
        STATE[$name]=done
        echo "[$(date)] $name: aggregate.json already exists, marking done"
    elif [ -f "$OUT_ROOT/$name/.firing" ]; then
        STATE[$name]=fired
        echo "[$(date)] $name: .firing sentinel present, marking fired"
    fi
done

ENV_PREFIX='export HF_TOKEN=$(grep "^HF_TOKEN=" /storage/home/sss6371/secure-code-rl-ictai/.env | cut -d= -f2) && export NVD_API_KEY=$(grep "^NVD_API_KEY=" /storage/home/sss6371/secure-code-rl-ictai/.env | cut -d= -f2) &&'
PATH_EXPORT='export PATH=/storage/home/sss6371/.local/share/codeql:'$REPO'/.venv/bin:/storage/work/sss6371/.conda/envs/sast/bin:$PATH'

fire_eval() {
    local name=$1
    local ckpt=$2
    mkdir -p "$OUT_ROOT/$name"
    local sentinel="$OUT_ROOT/$name/.firing"
    local jobname="ictai_he_v017_${name}"

    # Acquire a global cross-node fire-time lock. flock on shared NFS
    # serializes the squeue-check + sbatch step across all submit hosts,
    # so concurrent watchers can never both submit the same cell.
    (
        flock -x 200
        if [ -f "$sentinel" ]; then
            echo "[$(date)] $name: .firing sentinel present after lock, skipping"
            exit 0
        fi
        # Authoritative SLURM-side check inside the lock.
        if squeue -u $USER -o "%j" --noheader 2>/dev/null | grep -qx "$jobname"; then
            echo "[$(date)] $name: $jobname already in queue after lock, marking fired"
            echo "host=$(hostname) ts=$(date -u +%FT%TZ) reason=already-in-queue" > "$sentinel"
            exit 0
        fi
        local jid
        jid=$(sbatch --parsable \
            --account=szs339_cr_default \
            --partition=standard \
            --gres=gpu:a100:1 \
            --time=1800:00 \
            --mem=80G \
            --output="$REPO/logs/headline_eval_v017_${name}_%j.log" \
            --job-name="$jobname" \
            --wrap="$ENV_PREFIX $PATH_EXPORT && export HF_HOME=$HF_CACHE && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && cd $REPO && PYTHONPATH=src .venv/bin/python scripts/headline_eval.py --checkpoint $ckpt --name $name --eval-jsonl $EVAL_JSONL --output $OUT_ROOT --temperature 0.0 --max-new-tokens 512 --resume")
        echo "host=$(hostname) ts=$(date -u +%FT%TZ) jobid=$jid" > "$sentinel"
        echo "[$(date)] FIRED $name -> job $jid (ckpt=$ckpt)"
    ) 200>"$FIRE_LOCK"
}

iter=0
while true; do
    iter=$((iter+1))
    n_done=0
    n_fired=0
    n_pending=0
    for name in "${!CELLS[@]}"; do
        case "${STATE[$name]:-pending}" in
            done)  n_done=$((n_done+1)) ;;
            fired) n_fired=$((n_fired+1)) ;;
            pending) n_pending=$((n_pending+1)) ;;
        esac
    done
    echo "[$(date)] iter=$iter done=$n_done fired=$n_fired pending=$n_pending"

    if [ "$n_pending" -eq 0 ]; then
        echo "[$(date)] all cells fired or done; watcher exiting"
        break
    fi

    for name in "${!CELLS[@]}"; do
        [ "${STATE[$name]:-pending}" != "pending" ] && continue
        cell_dir="${CELLS[$name]}"
        # Prefer checkpoint-best (set by P0.4 trainer logic at last save)
        # else checkpoint-1000 (final step). Cell must also no longer be
        # actively training (no ictai_<name> in squeue).
        best=""
        if [ -e "$cell_dir/checkpoint-best" ]; then
            best="$cell_dir/checkpoint-best"
        elif [ -d "$cell_dir/checkpoint-1000" ]; then
            best="$cell_dir/checkpoint-1000"
        fi
        if [ -z "$best" ]; then
            continue  # checkpoint not yet saved
        fi
        # Guard: only fire if no training job is still running for this cell
        train_jobname_pattern=""
        case "$name" in
            arm_a_grpo|arm_a_grpo_rag|arm_a_ppo|arm_a_ppo_rag|arm_a_rloo|arm_a_rloo_rag|arm_a_raft|arm_a_raft_rag) train_jobname_pattern="ictai_$name" ;;
            arm_a_reweight|arm_a_uniform) train_jobname_pattern="ictai_$name" ;;
            bundle_*) train_jobname_pattern="ictai_${name/bundle_/bundle_}" ; train_jobname_pattern="ictai_${name#bundle_}" ; train_jobname_pattern="ictai_bundle_${name#bundle_}" ;;
            sft_*) train_jobname_pattern="ictai_sft_v017_${name#sft_}" ;;
            tierB_*) train_jobname_pattern="ictai_$name" ;;
            arm_a_grpo_uniform_rag) train_jobname_pattern="ictai_$name" ;;
        esac
        if [ -n "$train_jobname_pattern" ]; then
            # Fail-closed squeue guard. SLURM can transiently return empty
            # or non-zero. If any of N attempts says still-training, treat as
            # still-training. If ALL attempts fail or return empty, also
            # treat as still-training (do NOT fire) -- we would rather wait
            # one more iteration than fire a premature eval on a mid-train
            # checkpoint. See premature sft_r256 fire 2026-06-23.
            sq_attempts=0
            sq_still_training=0
            sq_any_success=0
            while [ $sq_attempts -lt 3 ]; do
                sq_out=$(squeue -u $USER -o "%j" --noheader 2>/dev/null)
                sq_rc=$?
                if [ $sq_rc -eq 0 ] && [ -n "$sq_out" ]; then
                    sq_any_success=1
                    if echo "$sq_out" | grep -qx "$train_jobname_pattern"; then
                        sq_still_training=1
                        break
                    fi
                fi
                sq_attempts=$((sq_attempts+1))
                sleep 2
            done
            if [ $sq_still_training -eq 1 ]; then
                continue  # confirmed still training
            fi
            if [ $sq_any_success -eq 0 ]; then
                echo "[$(date)] $name: squeue returned empty/error 3x in a row; skipping fire this iter"
                continue  # fail-closed: skip rather than fire
            fi
        fi
        fire_eval "$name" "$best"
        STATE[$name]=fired
    done

    sleep 300  # 5 min
done
