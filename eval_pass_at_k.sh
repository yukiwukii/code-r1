#!/bin/bash
# Evaluate a trained model with pass@k on the test set.
# Runs generation then pass@k evaluation.
#
# PBS example:
#   #PBS -q normal
#   #PBS -l select=1:ncpus=128:ngpus=4
#   #PBS -l walltime=02:00:00

# ─── CONFIG ────────────────────────────────────────────────────────────────────

# Model to evaluate (your trained checkpoint, or base model for baseline)
MODEL_PATH=./models/code-r1-12k-grpo/global_step_64   # adjust to your checkpoint

# Test data (prompt key must be 'prompt')
TEST_DATA=data/code-r1-12k/test_top5_dp.parquet

# Where to save generated responses
GEN_OUTPUT=/tmp/eval_generated.parquet

# Number of samples to generate per problem (higher → better pass@k coverage)
N_SAMPLES=16

# k values to report
KS="1 5 10 16"

# Where to save final JSON results
RESULTS_JSON=./eval_results.json

# Generation hyperparams
TEMPERATURE=1.0
TOP_P=0.95
TOP_K=-1
PROMPT_LEN=2048
RESPONSE_LEN=800
GPU_MEM_UTIL=0.8

# Evaluation
EVAL_WORKERS=8          # parallel threads for scoring
EVAL_TIMEOUT=120        # seconds per response before giving up

# ─── SINGULARITY / ENV ────────────────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES=0,1,2,3
export CODER1_EXEC=singularity
export VLLM_ATTENTION_BACKEND=XFORMERS

GPUS_PER_NODE=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')

set -euo pipefail
set -x

# ─── STEP 1: GENERATION ───────────────────────────────────────────────────────

echo "=== Step 1: Generating ${N_SAMPLES} responses per problem ==="
python3 -m verl.trainer.main_generation \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=$GPUS_PER_NODE \
    data.path=$TEST_DATA \
    data.prompt_key=prompt \
    data.n_samples=$N_SAMPLES \
    data.output_path=$GEN_OUTPUT \
    data.batch_size=32 \
    model.path=$MODEL_PATH \
    rollout.temperature=$TEMPERATURE \
    rollout.top_p=$TOP_P \
    rollout.top_k=$TOP_K \
    rollout.prompt_length=$PROMPT_LEN \
    rollout.response_length=$RESPONSE_LEN \
    rollout.gpu_memory_utilization=$GPU_MEM_UTIL \
    rollout.tensor_model_parallel_size=1 \
    rollout.name=vllm \
    2>&1 | tee eval_generation.log
echo "=== Generation done. Output: $GEN_OUTPUT ==="

# ─── STEP 2: PASS@K EVALUATION ────────────────────────────────────────────────

echo "=== Step 2: Computing pass@k ==="
python3 scripts/eval_pass_at_k.py \
    --input   "$GEN_OUTPUT" \
    --ks      $KS \
    --workers $EVAL_WORKERS \
    --timeout $EVAL_TIMEOUT \
    --output  "$RESULTS_JSON" \
    2>&1 | tee eval_scoring.log

echo "=== Evaluation complete. Results: $RESULTS_JSON ==="
