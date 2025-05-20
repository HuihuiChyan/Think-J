import os
import re
import json
import torch
import random
import argparse
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

from utils_args import build_parser


def process_output_cot(line, parsing="loose", log_num=1):

    # 查找thinking trace并且去除
    thinking_pattern = r"<think>.*?</think>"
    matches = re.findall(thinking_pattern, line, flags=re.DOTALL)

    # if len(matches) >= 1:
    #     # 去掉<think>标签及其内容
    #     line = re.sub(thinking_pattern, "", line, flags=re.DOTALL).strip()

    if parsing == "loose":
        output_pattern = {
            1: r"Response \(a\) is better",
            2: r"Response \(b\) is better"
        }
    else:
        output_pattern = {
            1: r"Therefore, Response \(a\) is better.$",
            2: r"Therefore, Response \(b\) is better.$"
        }
    for prediction, pattern in output_pattern.items():
        if re.search(pattern, line):
            return {"cot": line, "judgement": prediction}

    # global MAX_PARSE_LOG
    # if MAX_PARSE_LOG < log_num:
    #     print("######## Parsing fail! Response is: ########")
    #     print(line)
    #     MAX_PARSE_LOG += 1

    return {"cot": line.strip(), "judgement": 0}

def process_output_strength(line, parsing="loose", log_num=1):

    # 查找thinking trace并且去除
    thinking_pattern = r"<think>.*?</think>"
    matches = re.findall(thinking_pattern, line, flags=re.DOTALL)

    # if len(matches) >= 1:
    #     # 去掉<think>标签及其内容
    #     line = re.sub(thinking_pattern, "", line, flags=re.DOTALL).strip()

    score_pattern = r"Therefore, Response \((a|b)\) is better, and the strength is \[\[(\d+)\]\]\."
    
    # 使用正则表达式匹配输入字符串
    match = re.search(score_pattern, line)

    try:
        # 提取 preference 和 score
        preference = match.group(1)  # 提取偏好响应
        strength = int(match.group(2))  # 提取偏好强度并转换为整数
        if preference == "a":
            prediction = 1
        elif preference == "b":
            prediction = 2
        else:
            raise Exception("Preference not matched!")
        if strength not in {1, 2, 3}:
            raise Exception("Strength not matched!")
        return {"cot": line, "judgement": prediction, "strength": strength}

    except:

        score_pattern = r"Therefore, Response \((a|b)\) is better, and the strength is (\d+)\."

        # 使用正则表达式匹配输入字符串
        match = re.search(score_pattern, line)

        try:
            # 提取 preference 和 score
            preference = match.group(1)  # 提取偏好响应
            strength = int(match.group(2))  # 提取偏好强度并转换为整数
            if preference == "a":
                prediction = 1
            elif preference == "b":
                prediction = 2
            else:
                raise Exception("Preference not matched!")
            if strength not in {1, 2, 3}:
                raise Exception("Strength not matched!")
            
            line = re.sub(score_pattern, lambda match: f"Therefore, Response ({match.group(1)}) is better, and the strength is [[{match.group(2)}]].", line)

            return {"cot": line, "judgement": prediction, "strength": strength}

        except:
            print("Parsing failed! Input is")
            print(line)
            return {"cot": line, "judgement": 0, "strength": 0}


def infer_judge(lines, model, tokenizer, max_new_tokens, temperature, reverse=False, prompt_type="cot_judge_prompt", parsing="strict", log_num=1):
    global MAX_PARSE_LOG
    MAX_PARSE_LOG = 0

    if prompt_type == "strength_judge_prompt":
        from utils_prompts import strength_judge_prompt as judge_prompt
        process_output = process_output_strength
    elif prompt_type == "direct_judge_prompt":
        from utils_prompts import direct_judge_prompt as judge_prompt
        process_output = process_output_cot
    elif prompt_type == "cot_judge_prompt":
        from utils_prompts import cot_judge_prompt as judge_prompt
        process_output = process_output_cot
    
    if parsing == "loose":
        process_output = process_output_cot

    system_prompt = judge_prompt["system"]
    prompt_template = judge_prompt["user"]

    prompts = []
    betters = []
    for line in lines:
        if reverse:
            prompt = prompt_template.format(instruction=line['instruction'], response1=line['response2'], response2=line['response1'])
            betters.append(3-line['better']) # change 1 into 2, change 2 into 1
        else:
            prompt = prompt_template.format(instruction=line['instruction'], response1=line['response1'], response2=line['response2'])
            betters.append(line['better'])
                        
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
        prompts.append(model.llm_engine.tokenizer.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True))

    # Llama-3-8B-Instruct needs to set stop_token_id manually
    sampling_params = SamplingParams(max_tokens=max_new_tokens, temperature=temperature, n=1,
                                     stop_token_ids=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")])
    outputs = model.generate(prompt_token_ids=prompts, sampling_params=sampling_params)

    responses = [output.outputs[0].text for output in outputs]
    judgements = [process_output(r) for r in responses]

    return judgements, betters

if __name__ == "__main__":

    parser = build_parser()
    args = parser.parse_args()

    fin = open(args.input_file, "r")
    lines = json.load(fin)

    model = LLM(model=args.model_path, tensor_parallel_size=torch.cuda.device_count(), trust_remote_code=True, max_model_len=8192)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    random.seed(args.random_seed)

    judgements, betters = infer_judge(lines, model, tokenizer, args.max_new_tokens, args.temperature, prompt_type=args.prompt_type, reverse=False)

    if args.prompt_type == "strength_judge_prompt":
        from utils_prompts import strength_judge_prompt as judge_prompt
        process_output = process_output_strength
    elif args.prompt_type == "direct_judge_prompt":
        from utils_prompts import direct_judge_prompt as judge_prompt
        process_output = process_output_cot
    elif args.prompt_type == "cot_judge_prompt":
        from utils_prompts import cot_judge_prompt as judge_prompt
        process_output = process_output_cot

    new_lines = []
    for i in range(len(judgements)):
        if judgements[i]["judgement"] != 0:
            judgement_line = {
                                "system": judge_prompt["system"],
                                "instruction": judge_prompt["user"].format(
                                    instruction = lines[i]["instruction"],
                                    response1 = lines[i]["response1"],
                                    response2 = lines[i]["response2"],
                                ),
                                "input": "",
                                "output": judgements[i]["cot"],
                             }
            new_lines.append(judgement_line)
    sample_lines = random.sample(new_lines, k=3)
    for i in range(len(sample_lines)):
        sample_line = sample_lines[i]
        print(f"*************Sampled Line {i}*************")
        print(json.dumps(sample_line, indent=4))

    with open(args.output_file, "w") as fout:
        json.dump(new_lines, fout, indent=4)