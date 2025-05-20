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

def infer_critic(lines, model, tokenizer, max_new_tokens, temperature, is_positive="True", log_num=1):
    global MAX_PARSE_LOG
    MAX_PARSE_LOG = 0
    
    def process_output(line):
        output_pattern = {
            1: r"Response \(a\) is better.$",
            2: r"Response \(b\) is better.$"
        }
        for prediction, pattern in output_pattern.items():
            if re.search(pattern, line):
                return {"cot": line.strip(), "judgement": prediction}
        
        global MAX_PARSE_LOG        
        if MAX_PARSE_LOG < log_num:
            print("######## Parsing fail! Input is: ########")
            print(line)
            MAX_PARSE_LOG += 1

        return {"cot": line.strip(), "judgement": 0}

    from utils_prompts import critic_prompt

    system_prompt = critic_prompt["system"]
    prompt_template = critic_prompt["user"]

    prompts = []
    for line in lines:
        if (is_positive == "True" and line["better"] == 1) or \
           (is_positive == "False" and line["better"] == 2):
            prompt = prompt_template.format(
                instruction=line['instruction'], 
                response1=line['response1'], 
                response2=line['response2'],
                chosen="Response (a)",
                rejected="Response (b)",
            )
        else:
            prompt = prompt_template.format(
                instruction=line['instruction'], 
                response1=line['response1'], 
                response2=line['response2'],
                chosen="Response (b)",
                rejected="Response (a)",
            )
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
        prompts.append(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True))

    # Llama-3-8B-Instruct needs to set stop_token_id manually
    sampling_params = SamplingParams(max_tokens=max_new_tokens, temperature=temperature, 
                                     stop_token_ids=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")])
    outputs = model.generate(prompt_token_ids=prompts, sampling_params=sampling_params)
    responses = [output.outputs[0].text for output in outputs]
    judgements = [process_output(r) for r in responses]

    return judgements

if __name__ == "__main__":

    parser = build_parser()
    parser.add_argument("--is-positive", default="True", choices=("True", "False"))
    args = parser.parse_args()

    fin = open(args.input_file, "r")
    lines = json.load(fin)

    model = LLM(model=args.model_path, tensor_parallel_size=torch.cuda.device_count(), trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    random.seed(args.random_seed)

    judgements = infer_critic(lines, model, tokenizer, args.max_new_tokens, args.temperature, is_positive=args.is_positive)

    with open(args.output_file, "w") as fout:
        json.dump(judgements, fout, indent=4)