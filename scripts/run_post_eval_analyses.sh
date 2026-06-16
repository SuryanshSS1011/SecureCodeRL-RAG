#!/bin/bash
# Run every post-eval analysis script once headline_eval aggregates land.
# Idempotent: each script writes to a stable path and re-runs cheaply.
#
# Required state:
#   - aggregate.json present in each cell's eval-suite dir
#   - per_prompt_stream.jsonl present in each cell's eval-suite dir
#   - train_log.jsonl and eval_log.jsonl present in each cell's training
#     output dir
#
# Outputs all go to paper/results-data/ and paper/figures/.

set -euo pipefail

REPO=/storage/home/sss6371/secure-code-rl-ictai
SCRATCH=/scratch/sss6371/secure-code-rl-ictai-data
SWEEPS=$SCRATCH/sweeps
TRAIN_ROOT_V017=$SCRATCH/v0_1_7
TRAIN_ROOT_BUNDLE=$SCRATCH/v0_1_7_bundle
TRAIN_ROOT_HPARAMS=$SCRATCH/v0_1_7_hparams
PAPER=$REPO/paper

# Eval-suite path: this is what gets re-created when we re-fire headline_eval
# after the bug fixes. Update once we know the new suite name.
EVAL_SUITE=${EVAL_SUITE:-phase1_v0_1_7_headline_v2}

mkdir -p $PAPER/results-data $PAPER/figures

cd $REPO

echo "=== A3 / #246: per-CWE paired McNemar (RAG-on vs RAG-off) ==="
ARM_OFF=$SWEEPS/$EVAL_SUITE/arm_a_grpo/per_prompt_stream.jsonl
ARM_ON=$SWEEPS/$EVAL_SUITE/arm_a_grpo_rag/per_prompt_stream.jsonl
if [[ -f $ARM_OFF && -f $ARM_ON ]]; then
    PYTHONPATH=src .venv/bin/python scripts/per_cwe_mcnemar.py \
        --arm-off $ARM_OFF \
        --arm-on $ARM_ON \
        --output $PAPER/results-data/per_cwe_mcnemar.json \
        --tex-output $PAPER/results-data/per_cwe_mcnemar.tex
else
    echo "SKIP McNemar: streams not present yet ($EVAL_SUITE)"
fi

echo "=== multiseed summary (mean ± std across seed_1337 / seed_2024 / seed_42) ==="
SEED42=$SWEEPS/$EVAL_SUITE/arm_a_grpo_rag/aggregate.json
SEED1337=$SWEEPS/$EVAL_SUITE/seed_1337/aggregate.json
SEED2024=$SWEEPS/$EVAL_SUITE/seed_2024/aggregate.json
PATHS=""
[[ -f $SEED42 ]] && PATHS="$PATHS seed42:$SEED42"
[[ -f $SEED1337 ]] && PATHS="$PATHS seed1337:$SEED1337"
[[ -f $SEED2024 ]] && PATHS="$PATHS seed2024:$SEED2024"
if [[ -n $PATHS ]]; then
    PYTHONPATH=src .venv/bin/python scripts/multiseed_summary.py \
        --paths $PATHS \
        --output $PAPER/results-data/multiseed_headline.json
else
    echo "SKIP multiseed: no aggregates present"
fi

echo "=== done. Outputs under $PAPER/results-data/ and $PAPER/figures/ ==="
ls -la $PAPER/results-data/ 2>/dev/null | head -20
