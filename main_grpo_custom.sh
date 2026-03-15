#!/bin/bash
# Exercise 2 submission script - submit.sh
# Below, is the queue
#PBS -q normal
#PBS -j oe
#PBS -l select=1:ncpus=128:ngpus=4
#PBS -l walltime=01:00:00
#PBS -N grpo-run-custom
module load miniforge3
conda activate rl_cre
module load singularity

export CUDA_VISIBLE_DEVICES=0,1,2,3

cd /home/users/ntu/elim078/scratch/code-r1-yuki
# The config is optimized for 8xH200
set -x

if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    GPUS_PER_NODE=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
else
    GPUS_PER_NODE=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')
fi

# Jiawei's notes for 4xA100 PCIe (@Yifeng):
# - Becasue of PCIe, prefer gradient checkpointing over offloading
# - If offloading, prefer optimizer offloading (zero1) over param offloading
# - The code execution concurrency is $TOTAL_SAMPLES - nice to make it larger than $(nproc) to maximize CPUs
# - Try to make the #steps as long as possible: e.g., increasing epochs / reducing batches...
# - Set save_freq to a large number as I guess Colossus has little space left
# - If you are short of VRAM, consider removing reference policy. To do so, you need to go to
#    main_ppo.py:main_task - and comment "Role.RefPolicy..." in "role_worker_mapping = ".

# MAIN CONFIG
MAX_EPOCHS=1 #change it 
DATASET=code-r1-12k
MODEL_PATH=Qwen/Qwen2.5-7B-Instruct-1M
ROLLOUT_N_SAMPLE=16
ROLLOUT_N_QUERY=16
MICRO_BATCH_PER_GPU=1 # * GPUS_PER_NODE -> GLOBAL_BATCH_SIZE
GRAD_ACC_STEPS=4
GLOBAL_BATCH_SIZE=$(($(($GPUS_PER_NODE * $MICRO_BATCH_PER_GPU)) * $GRAD_ACC_STEPS))

# assert ROLLOUT_N_QUERY * ROLLOUT_N_SAMPLE % GLOBAL_BATCH_SIZE == 0
TOTAL_SAMPLES=$(( ROLLOUT_N_QUERY * ROLLOUT_N_SAMPLE ))
if (( TOTAL_SAMPLES % GLOBAL_BATCH_SIZE != 0 )); then
    echo "Error: (ROLLOUT_N_QUERY * ROLLOUT_N_SAMPLE) must be divisible by GLOBAL_BATCH_SIZE."
    echo "Currently, ${TOTAL_SAMPLES} is not divisible by ${GLOBAL_BATCH_SIZE}."
    exit 1
else
    echo "Assertion passed: ${TOTAL_SAMPLES} is divisible by ${GLOBAL_BATCH_SIZE}."
fi

export VLLM_ATTENTION_BACKEND=XFORMERS
export CODER1_EXEC=subprocess

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=data/$DATASET/train_top10_dp_double.parquet \
    data.val_files=data/$DATASET/test_top5_dp.parquet \
    data.train_batch_size=16 \
    data.max_prompt_length=2048 \
    data.max_response_length=800 \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$GLOBAL_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$MICRO_BATCH_PER_GPU \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=256 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.25 \
    actor_rollout_ref.rollout.n=$ROLLOUT_N_SAMPLE \
    actor_rollout_ref.ref.log_prob_micro_batch_size=256 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    algorithm.kl_ctrl.kl_coef=0.000 \
    trainer.critic_warmup=0 \
    trainer.logger=[] \
    trainer.project_name='code-r1-yuki' \
    trainer.experiment_name=${DATASET}-grpo \
    trainer.nnodes=1 \
    trainer.default_local_dir=./models/${DATASET}-grpo \
    trainer.n_gpus_per_node=$GPUS_PER_NODE \
    trainer.save_freq=64 \
    trainer.test_freq=16 \
    trainer.total_epochs=$MAX_EPOCHS \
    reward_model.reward_manager=prime $@ 2>&1 | tee grpo.log

#    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
#$ROLLOUT_N_QUERY
#train_batch_size= #16