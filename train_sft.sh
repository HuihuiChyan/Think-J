#!/bin/bash
set +x

workdir="./LLaMA-Factory"
cd $workdir

WORLD_SIZE=8

# 产出的Model信息
OUTPUT_MODEL=$1

# local_file或者oss挂载文件
INPUT=$2

INPUT_MODEL=$3

if [[ $INPUT_MODEL == *"Llama"* ]]; then
    PROMPT_TEMPLATE="llama3"
fi

if [[ $INPUT_MODEL == *"Qwen"* ]]; then
    PROMPT_TEMPLATE="qwen"
fi

if [[ $INPUT_MODEL == *"Mistral"* ]]; then
    PROMPT_TEMPLATE="mistral"
fi

echo "PROMPT_TEMPLATE: ${PROMPT_TEMPLATE}"

LR=5e-6

args="--stage sft \
      --model_name_or_path=$INPUT_MODEL \
      --do_train \
      --file_name=${INPUT} \
      --system=system \
      --prompt=instruction \
      --query=input \
      --response=output \
      --template=${PROMPT_TEMPLATE} \
      --finetuning_type full \
      --output_dir=${OUTPUT_MODEL} \
      --overwrite_cache \
      --flash_attn fa2 \
      --per_device_train_batch_size 1 \
      --gradient_accumulation_steps 8 \
      --lr_scheduler_type cosine \
      --logging_steps 1 \
      --save_strategy epoch \
      --num_train_epochs 3 \
      --learning_rate=$LR \
      --cutoff_len=4096 \
      --preprocessing_num_workers=8 \
      --dataloader_num_workers=4 \
      --plot_loss \
      --deepspeed=scripts/ds_zero3.json \
      --bf16 \
      --save_only_model \
      --report_to=none"

accelerate launch --main_process_port 29501 --multi_gpu src/train.py ${args}