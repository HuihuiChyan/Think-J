# Think-J

This repository is for Think-J: Learning to Think for Generative LLM-as-a-Judge


# ⚡️ Usage

### Preparation

Our implementation is based on [Llama-Factory](https://github.com/hiyouga/LLaMA-Factory) and [verl](https://github.com/volcengine/verl). Therefore, you can refer to their repository to prepare the environment.

For initialization data, we use the preference data from [Skywork](https://huggingface.co/datasets/Skywork/Skywork-Reward-Preference-80K-v0.2), and we present the processed data with thinking trace annotation in `data` directory.

For thinking optimiazation, we use the preference data from [Helpsteer2](https://huggingface.co/datasets/nvidia/HelpSteer2) and HH-RLHF(https://huggingface.co/datasets/Anthropic/hh-rlhf). To achieve the strength annotation, we leverage the preference annotation from [hh-rlhf-strength-cleaned](https://huggingface.co/datasets/fnlp/hh-rlhf-strength-cleaned). Please download them and present them in your data directory.

### Judgment Thinking Initilization

Based on the LIMJ707 data, initialize the judge with thinking ability based on Supervised Fine-tuning.
```shell
DATASET="helpsteer2"
MODEL="Qwen2.5-7B-Instruct"
PROMPT=strength
DATA_DIR=/path/to/your/data
MODEL_DIR=/path/to/your/model
R1JUDGE_VER="judge-skywork707-${PROMPT}-671BR1"
R1CRITIC_VER="critic-skywork707-${PROMPT}-671BR1"
BASE_PATH=/data/cpfs_0/bumblebee/TuningFactoryModels/${MODEL}
MOS_REPO=${MODEL}-STaR

DATA=skywork707-${PROMPT}-671BR1-judgement.json
bash train_sft.sh ${MODEL_DIR}/${MOS_REPO}-${R1JUDGE_VER} $DATA_DIR/${DATASET}/$DATA ${MODEL_DIR}/${MODEL}

DATA=skywork707-${PROMPT}-671BR1-critique.json
bash train_sft.sh ${MODEL_DIR}/${MOS_REPO}-${R1CRITIC_VER} $DATA_DIR/${DATASET}/$DATA ${MODEL_DIR}/${MODEL}
```

### Judgment Thinking Optimization

Based on preference data, optimize the thinking ability of the judge based on either offline RL or online RL. 

For offline RL, run the following scripts to firstly construct preference pairs and then perform training.
```shell
DATASET="helpsteer2"
MODEL="Qwen2.5-7B-Instruct"
PROMPT=strength
DATA_DIR=/path/to/your/data
MODEL_DIR=/path/to/your/model
R1JUDGE_VER="judge-skywork707-${PROMPT}-671BR1"
R1CRITIC_VER="critic-skywork707-${PROMPT}-671BR1"
MOS_REPO=${MODEL}-STaR

python3.10 star/infer_judge_create_prompt.py \
    --model-path ${MODEL_DIR}/${MOS_REPO}-${R1JUDGE_VER} \
    --input-file ${DATA_DIR}/${DATASET}/${DATASET}-infer.json \
    --output-file ${DATA_DIR}/${DATASET}/${DATASET}-${MODEL}-judgement-jud.json \
    --prompt-type ${PROMPT}_judge_prompt

python3.10 star/infer_critic_create_prompt.py \
    --model-path ${MODEL_DIR}/${MOS_REPO}-${R1CRITIC_VER} \
    --input-file ${DATA_DIR}/${DATASET}/${DATASET}-infer.json \
    --is-positive True \
    --output-file ${DATA_DIR}/${DATASET}/${DATASET}-${MODEL}-judgement-pos.json \
    --prompt-type ${PROMPT}_judge_prompt

python3.10 star/infer_critic_create_prompt.py \
    --model-path ${MODEL_DIR}/${MOS_REPO}-${R1CRITIC_VER} \
    --input-file ${DATA_DIR}/${DATASET}/${DATASET}-infer.json \
    --is-positive False \
    --output-file ${DATA_DIR}/${DATASET}/${DATASET}-${MODEL}-judgement-neg.json \
    --prompt-type ${PROMPT}_judge_prompt

python3.10 star/create_judgement_pairs.py \
    ${DATA_DIR}/${DATASET}/${DATASET}-${MODEL}-judgement-jud.json \
    ${DATA_DIR}/${DATASET}/${DATASET}-${MODEL}-judgement-pos.json \
    ${DATA_DIR}/${DATASET}/${DATASET}-${MODEL}-judgement-neg.json \
    ${DATA_DIR}/${DATASET}/${DATASET}-${MODEL}-judgement.json \
    ${DATA_DIR}/${DATASET}/${DATASET}-${MODEL}-judgement-sft.json

DATA=${DATASET}-${MODEL}-judgement.json
JUDGE_MODEL=${MOS_REPO}-${DATASET}-judge-dpo-${PROMPT}-671BR1
bash train_dpo.sh ${MODEL_DIR}/${JUDGE_MODEL} $DATA_DIR/${DATASET}/$DATA ${MODEL_DIR}/${MOS_REPO}-${R1JUDGE_VER}
```

For online RL, run the following scripts to perform GRPO training.

```shell
DATASET="helpsteer2"
MODEL="Qwen2.5-7B-Instruct"
PROMPT=strength
DATA_DIR=/path/to/your/data
MODEL_DIR=/path/to/your/model
R1JUDGE_VER="judge-skywork707-${PROMPT}-671BR1"
R1CRITIC_VER="critic-skywork707-${PROMPT}-671BR1"
MOS_REPO=${MODEL}-STaR

DATA=${DATASET}-${PROMPT}-ppo.parquet
DATA_TEST=${DATASET}-${PROMPT}-ppo-test.parquet
JUDGE_MODEL=${MOS_REPO}-${DATASET}-judge-grpo-${PROMPT}-671BR1
bash train_grpo.sh ${MODEL_DIR}/${JUDGE_MODEL} $DATA_DIR/${DATASET}/${DATA} $DATA_DIR/${DATASET}/${DATA_TEST} ${MODEL_DIR}/${MOS_REPO}-${R1JUDGE_VER} 8 reward_${PROMPT}

FINAL_STEP=$(cat ${MODEL_DIR}/${JUDGE_MODEL}/latest_checkpointed_iteration.txt)

python3.10 verl/scripts/model_merger.py \
    --local_dir ${MODEL_DIR}/${JUDGE_MODEL}/global_step_${FINAL_STEP}/actor
```

### Evaluation on RewardBench

We mainly conduct evaluation on RewardBench. Run the following script to test your generative judge on RewardBench.
```shell
PROMPT=strength
JUDGE_MODEL=Qwen2.5-7B-Instruct-STaR-helpsteer2-judge-grpo-strength-671BR1
MODEL_PATH=/path/to/your/model

python3.10 star/eval_reward_bench.py \
    --model-path ${MODEL_DIR}/{JUDGE_MODEL} \
    --prompt-type ${PROMPT}_judge_prompt \
```

# Acknowledge

This repository is built on [Llama-Factory](https://github.com/hiyouga/LLaMA-Factory) and [verl](https://github.com/volcengine/verl). Many thanks to their excellent work!