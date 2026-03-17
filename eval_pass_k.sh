#!/bin/bash

#PBS -q normal
#PBS -j oe
#PBS -l select=1:ncpus=128:ngpus=4
#PBS -l walltime=3:00:00
#PBS -N pass-k-eval

module load miniforge3
conda activate rl_cre
module load singularity

cd /home/users/ntu/elim078/scratch/code-r1-yuki

# MAIN CONFIG 
MODEL_PATH=models/code-r1-12k-grpo/global_step_6/actor # Model to evaluate
TEST_DATA=data/code-r1-12k/test_top5_dp.parquet # Test data (prompt key must be 'prompt')
GEN_OUTPUT=data/eval_generated.parquet # Save generated output
N_SAMPLES=16 # No. of rollout
KS="1 5 10 16" # k values to report
RESULTS_JSON=data/eval_results.json # Final report output

# GENERATION HYPERPARAMS
TEMPERATURE=1.0
TOP_P=0.95
TOP_K=-1
PROMPT_LEN=2048
RESPONSE_LEN=800
GPU_MEM_UTIL=0.25

# PARALLEL EVAL (SANDBOX STUFF)
EVAL_WORKERS=8          # parallel threads for scoring
EVAL_TIMEOUT=120        # seconds per response before giving up

# GPU SETUPS
export CUDA_VISIBLE_DEVICES=0,1,2,3
export CODER1_EXEC=singularity
export VLLM_ATTENTION_BACKEND=XFORMERS

GPUS_PER_NODE=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')

set -euo pipefail
set -x

# Actually need to merge first for some reason
# Has to do with huggingface stuff
python3 scripts/model_merger.py \
    --local_dir ${MODEL_PATH}  


echo "Step 1: Generating ${N_SAMPLES} responses per problem"
python3 -m verl.trainer.main_generation \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=$GPUS_PER_NODE \
    data.path=$TEST_DATA \
    data.prompt_key=prompt \
    data.n_samples=$N_SAMPLES \
    data.output_path=$GEN_OUTPUT \
    data.batch_size=32 \
    model.path=${MODEL_PATH}/huggingface \
    rollout.temperature=$TEMPERATURE \
    rollout.top_p=$TOP_P \
    rollout.top_k=$TOP_K \
    rollout.prompt_length=$PROMPT_LEN \
    rollout.response_length=$RESPONSE_LEN \
    rollout.gpu_memory_utilization=$GPU_MEM_UTIL \
    rollout.tensor_model_parallel_size=4 \
    rollout.name=vllm \
    2>&1 | tee eval_generation.log
echo "Generation done. Output: $GEN_OUTPUT"


echo "Step 2: Computing pass@k"
python3 scripts/eval_pass_k.py \
    --input   "$GEN_OUTPUT" \
    --ks      $KS \
    --workers $EVAL_WORKERS \
    --timeout $EVAL_TIMEOUT \
    --output  "$RESULTS_JSON" \
    2>&1 | tee eval_scoring.log

echo "Evaluation complete. Results: $RESULTS_JSON"