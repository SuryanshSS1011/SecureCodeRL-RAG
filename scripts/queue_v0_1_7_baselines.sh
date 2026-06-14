#!/bin/bash
# Re-baseline 9 models against v0.1.7 (1,529 eval prompts).
#
# Each baseline gets a 2-stage afterany chain. Stage 1 runs fresh; stage 2
# fires if stage 1 timed out and uses --resume-from-stream to skip prompts
# already scored. Stream flush in run_baseline_sweep.py (commit 93f8227)
# preserves per-prompt records on SLURM timeout.
#
# Per-baseline wall budget: ~14h observed for 1.5B at 0.03 prompts/s;
# 7B variants run ~16-19h. 24h chain stage gives margin. Two stages = 48h
# total which is well below the 14-day partition ceiling.
#
# Output layout:
#   /scratch/.../sweeps/phase1_v0_1_7/
#     <baseline-name>/
#       sweep_summary.json
#       per_prompt_stream.jsonl  (streamed; resume reads this)

set -euo pipefail

REPO=/storage/home/sss6371/secure-code-rl-ictai
SCRATCH_ROOT=/scratch/sss6371/secure-code-rl-ictai-data
EVAL=$SCRATCH_ROOT/build/v0.1.7/eval_prompts.jsonl
OUT_ROOT=$SCRATCH_ROOT/sweeps/phase1_v0_1_7
LOGS=$REPO/logs

if [[ ! -f "$EVAL" ]]; then
    echo "ERROR: $EVAL not found." >&2
    exit 1
fi

mkdir -p "$OUT_ROOT" "$SCRATCH_ROOT/hf_cache"

BASELINES=(
    qwen2.5-coder-1.5b
    qwen2.5-coder-3b
    qwen2.5-coder-7b
    starcoder2-3b
    sven-codegen-2.7b
    seccoderx-qwen2.5-coder-3b
    seccoderx-qwen2.5-coder-7b
)

# Per-stage walltime. 24h = 1440 min. Plenty for 1,529-prompt eval.
WALLTIME=${WALLTIME:-1440}
N_CHAIN=${N_CHAIN:-2}

for b in "${BASELINES[@]}"; do
    cfg="$OUT_ROOT/${b}_config.json"
    cat > "$cfg" <<EOF
{
  "baselines": [
    {"name": "${b}", "baseline": "${b}"}
  ]
}
EOF

    dep=""
    for i in $(seq 1 "$N_CHAIN"); do
        # Always pass --resume-from-stream. The flag is idempotent: it
        # no-ops on a fresh run (nothing to skip) but on stage 2+ it picks
        # up exactly where stage 1 left off after a SLURM timeout. This
        # also lets us re-launch after losing a previous run without
        # restarting from prompt 0 (per FINDINGS_LOG 2026-06-15 recovery
        # from the 11h-walltime kill).
        resume_flag="--resume-from-stream"
        jid=$(sbatch --parsable $dep \
            --account=szs339_cr_default \
            --partition=standard \
            --gres=gpu:a40:1 \
            --time=${WALLTIME}:00 \
            --mem=80G \
            --output="$LOGS/phase1_v0_1_7_${b}_%j.log" \
            --job-name="ictai_p1_v017_${b}" \
            --wrap="set -a; [ -f $REPO/.env ] && . $REPO/.env; set +a; export HF_HOME=$SCRATCH_ROOT/hf_cache/${b} && mkdir -p \$HF_HOME && cd $REPO && PYTHONPATH=src .venv/bin/python scripts/run_baseline_sweep.py --eval-jsonl $EVAL --config $cfg --output $OUT_ROOT/${b} --oracle-kind real --max-new-tokens 512 $resume_flag")
        echo "  $b stage $i/$N_CHAIN: job $jid"
        # afterany so timeouts trigger the resume stage.
        dep="--dependency=afterany:$jid"
    done
done

echo
echo "Submitted ${#BASELINES[@]} baseline chains ($N_CHAIN stages each, ${WALLTIME} min per stage)."
echo
echo "Watch with:"
echo "  squeue --user=\$USER --name=ictai_p1_v017*"
echo "  tail -f $LOGS/phase1_v0_1_7_*.log"
echo
echo "Results under: $OUT_ROOT/"
echo
echo "Once all done, cross-baseline comparison:"
