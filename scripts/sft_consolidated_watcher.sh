#!/bin/bash
# Consolidated SFT watcher v2 (idempotent)
# Detects existing in-flight jobs on startup to avoid duplicate fires.
set -u
LOG=/storage/home/sss6371/secure-code-rl-ictai/logs/sft_consolidated_watcher.log
exec >> "$LOG" 2>&1
echo "[$(date)] watcher v2 start, pid=$$"

EVAL_BASE=/scratch/sss6371/secure-code-rl-ictai-data/sweeps/v0_1_7_sft_only_evals
EVAL_JSONL=/scratch/sss6371/secure-code-rl-ictai-data/build/v0.1.7/eval_prompts.jsonl
ENV_PREFIX='set -a; [ -f /storage/home/sss6371/secure-code-rl-ictai/.env ] && . /storage/home/sss6371/secure-code-rl-ictai/.env; set +a;'

declare -A STATE

# Detect any existing running/queued jobs and set initial state idempotently.
init_state() {
  for r in 16 32 64 128 256 512; do
    STATE[$r]=pending
  done
  # r=16 extend already DONE per disk evidence
  STATE[16]=extend_done
  # Check for any existing eval job (either naming convention)
  for r in 16 32 64 128 256 512; do
    for jn in "ictai_v3_sft_eval_r${r}" "ictai_v3_sft_r${r}_eval"; do
      if squeue -u $USER -o "%j" --noheader 2>/dev/null | grep -qx "$jn"; then
        STATE[$r]=eval_fired
        echo "[$(date)] detected existing eval job $jn — STATE[$r]=eval_fired"
      fi
    done
    # Check for extension job already running
    if squeue -u $USER -o "%j" --noheader 2>/dev/null | grep -qx "ictai_v3_sft_r${r}_extend"; then
      if [ "${STATE[$r]}" = "pending" ]; then
        STATE[$r]=extend_fired
        echo "[$(date)] detected existing extend job ictai_v3_sft_r${r}_extend — STATE[$r]=extend_fired"
      fi
    fi
  done

  # Fullft state (independent of LoRA ranks; does NOT gate C2 warm-start pick).
  FULLFT_STATE=pending
  if squeue -u $USER -o "%j" --noheader 2>/dev/null | grep -qx "ictai_v3_sft_fullft_eval"; then
    FULLFT_STATE=eval_fired
    echo "[$(date)] detected existing eval job ictai_v3_sft_fullft_eval — FULLFT_STATE=eval_fired"
  elif squeue -u $USER -o "%j" --noheader 2>/dev/null | grep -qx "ictai_v3_sft_fullft"; then
    FULLFT_STATE=train_running
    echo "[$(date)] detected existing fullft training — FULLFT_STATE=train_running"
  fi
  # If fullft eval has already produced an aggregate.json, mark eval_done.
  if [ -f "$EVAL_BASE/sft_fullft/aggregate.json" ] && [ "$FULLFT_STATE" = "pending" ]; then
    FULLFT_STATE=eval_done
  fi
}

fire_extension() {
  local r=$1 ckpt=$2
  # Idempotency: skip if extension job is already queued/running.
  if squeue -u $USER -o "%j" --noheader 2>/dev/null | grep -qx "ictai_v3_sft_r${r}_extend"; then
    echo "[$(date)] SKIP fire_extension r=$r: ictai_v3_sft_r${r}_extend already in squeue"
    return 1
  fi
  echo "[$(date)] firing extension for r=$r from $ckpt"
  sbatch --parsable --account=szs339_cr_default --partition=standard \
    --gres=gpu:a100:1 --time=1800:00 --mem=80G \
    --output=/storage/home/sss6371/secure-code-rl-ictai/logs/v3_sft_r${r}_extend_%j.log \
    --job-name=ictai_v3_sft_r${r}_extend \
    --wrap="$ENV_PREFIX export HF_HOME=/scratch/sss6371/secure-code-rl-ictai-data/hf_cache/sft_only_r${r} && cd /storage/home/sss6371/secure-code-rl-ictai && PYTHONPATH=src .venv/bin/python scripts/train_sft.py --sft-pairs /scratch/sss6371/secure-code-rl-ictai-data/build/v0.1.7/train_sft_pairs_aligned.jsonl --output /scratch/sss6371/secure-code-rl-ictai-data/v0_1_7_sft_only_r${r}/adapter --total-steps 20000 --batch-size 4 --learning-rate 2e-5 --lora-r ${r} --lora-alpha $((2*r)) --save-every 100 --val-fraction 0.1 --val-every 100 --resume-from $ckpt"
}

fire_eval() {
  local r=$1
  if eval_any_running $r; then
    echo "[$(date)] SKIP fire_eval r=$r: eval job already in squeue"
    return 1
  fi
  local outdir
  if [ "$r" = "16" ]; then
    outdir=/scratch/sss6371/secure-code-rl-ictai-data/v0_1_7_sft_only_fresh10k/adapter
  else
    outdir=/scratch/sss6371/secure-code-rl-ictai-data/v0_1_7_sft_only_r${r}/adapter
  fi
  local best=$outdir/checkpoint-best
  local evdir=$EVAL_BASE/sft_r${r}
  mkdir -p $evdir
  echo "[$(date)] firing headline_eval for r=$r against $best"
  sbatch --parsable --account=szs339_cr_default --partition=standard \
    --gres=gpu:a100:1 --time=900:00 --mem=80G \
    --output=/storage/home/sss6371/secure-code-rl-ictai/logs/v3_sft_r${r}_eval_%j.log \
    --job-name=ictai_v3_sft_r${r}_eval \
    --wrap="$ENV_PREFIX export HF_HOME=/scratch/sss6371/secure-code-rl-ictai-data/hf_cache/sft_eval_r${r} && cd /storage/home/sss6371/secure-code-rl-ictai && PYTHONPATH=src .venv/bin/python scripts/headline_eval.py --checkpoint $best --eval-jsonl $EVAL_JSONL --output $evdir --name sft_r${r}"
}

job_state() {
  local jname=$1
  if squeue -u $USER -o "%j" --noheader 2>/dev/null | grep -qx "$jname"; then
    echo "RUNNING"
  else
    echo "NONE"
  fi
}

eval_any_running() {
  local r=$1
  for jn in "ictai_v3_sft_eval_r${r}" "ictai_v3_sft_r${r}_eval"; do
    if squeue -u $USER -o "%j" --noheader 2>/dev/null | grep -qx "$jn"; then return 0; fi
  done
  return 1
}

fire_eval_fullft() {
  if squeue -u $USER -o "%j" --noheader 2>/dev/null | grep -qx "ictai_v3_sft_fullft_eval"; then
    echo "[$(date)] SKIP fire_eval_fullft: already in squeue"
    return 1
  fi
  local best=/scratch/sss6371/secure-code-rl-ictai-data/v0_1_7_sft_only_fullft/adapter/checkpoint-best
  local evdir=$EVAL_BASE/sft_fullft
  mkdir -p $evdir
  echo "[$(date)] firing headline_eval for fullft against $best"
  sbatch --parsable --account=szs339_cr_default --partition=standard \
    --gres=gpu:a100:1 --time=900:00 --mem=80G \
    --output=/storage/home/sss6371/secure-code-rl-ictai/logs/v3_sft_fullft_eval_%j.log \
    --job-name=ictai_v3_sft_fullft_eval \
    --wrap="$ENV_PREFIX export HF_HOME=/scratch/sss6371/secure-code-rl-ictai-data/hf_cache/sft_eval_fullft && cd /storage/home/sss6371/secure-code-rl-ictai && PYTHONPATH=src .venv/bin/python scripts/headline_eval.py --checkpoint $best --eval-jsonl $EVAL_JSONL --output $evdir --name sft_fullft"
}

init_state
echo "[$(date)] initial states: r16=${STATE[16]} r32=${STATE[32]} r64=${STATE[64]} r128=${STATE[128]} r256=${STATE[256]} r512=${STATE[512]} fullft=$FULLFT_STATE"

ITER=0
while true; do
  ITER=$((ITER+1))
  echo "[$(date)] iter=$ITER states: r16=${STATE[16]} r32=${STATE[32]} r64=${STATE[64]} r128=${STATE[128]} r256=${STATE[256]} r512=${STATE[512]} fullft=$FULLFT_STATE"

  # Fullft state machine (independent of LoRA ranks; does NOT gate C2).
  # train_running → eval_fired when training job exits and checkpoint-best
  # is on disk; eval_fired → eval_done when the eval job exits.
  case "$FULLFT_STATE" in
    pending|train_running)
      stf=$(job_state ictai_v3_sft_fullft)
      if [ "$stf" = "NONE" ]; then
        ckpt=/scratch/sss6371/secure-code-rl-ictai-data/v0_1_7_sft_only_fullft/adapter/checkpoint-best
        if [ -e "$ckpt" ]; then
          fire_eval_fullft && FULLFT_STATE=eval_fired
        else
          FULLFT_STATE=train_running
        fi
      else
        FULLFT_STATE=train_running
      fi
      ;;
    eval_fired)
      ste=$(job_state ictai_v3_sft_fullft_eval)
      if [ "$ste" = "NONE" ]; then
        # Race guard: sbatch return + squeue listing have ~10s lag, so
        # NONE alone can fire spuriously moments after sbatch returns.
        # Require aggregate.json on disk before declaring eval_done.
        if [ -f "$EVAL_BASE/sft_fullft/aggregate.json" ]; then
          FULLFT_STATE=eval_done
          echo "[$(date)] fullft eval complete"
        else
          # Job is gone but no aggregate => job crashed or never started.
          # Reset to allow re-fire. (fire_eval_fullft is idempotent.)
          echo "[$(date)] fullft eval job NONE but no aggregate.json; will retry"
          FULLFT_STATE=train_running
        fi
      fi
      ;;
  esac


  # For each rank, advance state. extend_done → eval_fired only if no eval job already in flight.
  for r in 16 32 64 128 256 512; do
    case "${STATE[$r]}" in
      extend_done)
        if eval_any_running $r; then
          STATE[$r]=eval_fired
          echo "[$(date)] r=$r: eval already running, advancing to eval_fired"
        else
          fire_eval $r && STATE[$r]=eval_fired
        fi
        ;;
      pending)
        if [ "$r" = "16" ]; then continue; fi  # r=16 starts at extend_done
        # Check original training job
        st=$(job_state ictai_v3_sft_r${r})
        if [ "$st" = "NONE" ]; then
          # Original done. If r=32, the extend was already running; let extend_fired branch handle it.
          # If r=64 or r=128, fire extension.
          if [ "$r" = "32" ]; then
            # r=32 extend may already be running
            stx=$(job_state ictai_v3_sft_r32_extend)
            if [ "$stx" = "RUNNING" ]; then
              STATE[$r]=extend_fired
            else
              STATE[$r]=extend_done
            fi
          else
            ckpt=/scratch/sss6371/secure-code-rl-ictai-data/v0_1_7_sft_only_r${r}/adapter/checkpoint-10000
            if [ -d "$ckpt" ]; then
              fire_extension $r "$ckpt" && STATE[$r]=extend_fired
            fi
          fi
        fi
        ;;
      extend_fired)
        st=$(job_state ictai_v3_sft_r${r}_extend)
        if [ "$st" = "NONE" ]; then
          STATE[$r]=extend_done
        fi
        ;;
    esac
  done

  # Done check: all 6 LoRA evals must have COMPLETED (not just fired)
  # before we pick the winner. Eval F1 is the criterion, not val_loss.
  all_complete=1
  for r in 16 32 64 128 256 512; do
    if [ "${STATE[$r]}" != "eval_fired" ]; then all_complete=0; break; fi
    if squeue -u $USER -o "%j" --noheader 2>/dev/null | grep -qx "ictai_v3_sft_eval_r${r}"; then all_complete=0; break; fi
    if squeue -u $USER -o "%j" --noheader 2>/dev/null | grep -qx "ictai_v3_sft_r${r}_eval"; then all_complete=0; break; fi
    # Race guard: require aggregate.json on disk (sbatch/squeue lag).
    if [ ! -f "$EVAL_BASE/sft_r${r}/aggregate.json" ]; then all_complete=0; break; fi
  done
  if [ "$all_complete" = "1" ] && [ "${LADDER_DONE:-0}" != "1" ]; then
    echo "[$(date)] all 6 LoRA SFT evals complete, picking winner by eval F1"
    best_r=""
    best_f1=-1
    for r in 16 32 64 128 256 512; do
      agg=$EVAL_BASE/sft_r${r}/aggregate.json
      f1=$(python3 -c "import json,sys
try:
  d=json.load(open('$agg'))
  m=d.get('metrics',d)
  print(m.get('func_sec_at_1__compiles_and_has_tests',m.get('func_sec_at_1',-1)))
except Exception as e:
  print(-1)" 2>/dev/null)
      echo "  r=$r eval_f1=$f1"
      if [ -n "$f1" ] && awk -v a=$f1 -v b=$best_f1 'BEGIN{exit !(a>b)}'; then
        best_f1=$f1
        best_r=$r
      fi
    done
    echo "[$(date)] WINNER: r=$best_r eval_f1=$best_f1"
    if [ -n "$best_r" ]; then
      if [ "$best_r" = "16" ]; then
        adapter=/scratch/sss6371/secure-code-rl-ictai-data/v0_1_7_sft_only_fresh10k/adapter/checkpoint-best
      else
        adapter=/scratch/sss6371/secure-code-rl-ictai-data/v0_1_7_sft_only_r${best_r}/adapter/checkpoint-best
      fi
      echo "[$(date)] firing Tier C2: GRPO+RAG warm-started from r=$best_r adapter $adapter"
      sbatch --parsable --account=szs339_cr_default --partition=standard \
        --gres=gpu:a100:1 --time=2400:00 --mem=80G \
        --output=/storage/home/sss6371/secure-code-rl-ictai/logs/v3_tierC2_sft_warm_r${best_r}_%j.log \
        --job-name=ictai_v3_tierC2_sft_warm_r${best_r} \
        --wrap="$ENV_PREFIX export HF_HOME=/scratch/sss6371/secure-code-rl-ictai-data/hf_cache/tierC2 && cd /storage/home/sss6371/secure-code-rl-ictai && PYTHONPATH=src .venv/bin/python scripts/train_method.py --config configs/headline_grpo_rag.yaml --output /scratch/sss6371/secure-code-rl-ictai-data/sweeps/phase1_v0_1_7/tierC2_sft_warm_r${best_r} --warm-start-adapter $adapter"
    fi
    LADDER_DONE=1
  fi

  # Terminal: LoRA ladder + C2 fired AND fullft eval done.
  if [ "${LADDER_DONE:-0}" = "1" ] && [ "$FULLFT_STATE" = "eval_done" ]; then
    fullft_agg=$EVAL_BASE/sft_fullft/aggregate.json
    fullft_f1=$(python3 -c "import json
try:
  d=json.load(open('$fullft_agg'))
  m=d.get('metrics',d)
  print(m.get('func_sec_at_1__compiles_and_has_tests',m.get('func_sec_at_1',-1)))
except Exception:
  print(-1)" 2>/dev/null)
    echo "[$(date)] FULLFT REFERENCE: eval_f1=$fullft_f1"
    echo "[$(date)] all done; watcher exiting"
    break
  fi

  sleep 300
done
