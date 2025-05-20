#!/bin/bash
set +x

workdir="./verl"
cd $workdir

WORLD_SIZE=2

# 产出的Model信息
OUTPUT_MODEL=$1

# local_file或者oss挂载文件
INPUT=$2
INPUT_TEST=$3

INPUT_MODEL=$4

BATCH_SIZE=$5

REWARD_FUNC=$6

entry_file="verl/trainer/main_ppo.py"

python3.10 scripts/change_yaml.py \
    --file ./verl/trainer/config/ppo_trainer.yaml.hist \
    --output ./verl/trainer/config/ppo_trainer.yaml \
    --key-value \
    algorithm.adv_estimator=grpo \
    data.train_files=$INPUT \
    data.val_files=$INPUT_TEST \
    data.train_batch_size=1024 \
    data.max_prompt_length=1024 \
    data.max_response_length=3072 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=$INPUT_MODEL \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$BATCH_SIZE \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$BATCH_SIZE \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$BATCH_SIZE \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    custom_reward_function.path=verl/utils/reward_score/$REWARD_FUNC.py \
    custom_reward_function.name=compute_score \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.default_local_dir=$OUTPUT_MODEL \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.project_name='STaR-Judge' \
    trainer.experiment_name=$OUTPUT_MODEL \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=$WORLD_SIZE \
    trainer.save_freq=10 \
    trainer.test_freq=-1 \
    trainer.total_epochs=10 \
    trainer.val_before_train=False 

python3.10 ${entry_file} ${args}