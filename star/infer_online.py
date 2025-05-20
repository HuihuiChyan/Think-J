import os
import re
import json
import time
import random
import openai
import argparse
import multiprocessing
import timeout_decorator
from functools import partial
from utils_args import build_parser
from utils_prompts import cot_judge_prompt as judge_prompt
from utils_prompts import cot_critic_prompt as critic_prompt

# def process_output(line):

#     # 查找thinking trace并且去除
#     thinking_pattern = r"<think>.*?</think>"
#     matches = re.findall(thinking_pattern, line, flags=re.DOTALL)

#     # if len(matches) >= 1:
#     #     # 去掉<think>标签及其内容
#     #     line = re.sub(thinking_pattern, "", line, flags=re.DOTALL).strip()

#     score_pattern = r"Therefore, Response \((a|b)\) is better, and the strength is \[\[(\d+)\]\]\."
    
#     # 使用正则表达式匹配输入字符串
#     match = re.search(score_pattern, line)

#     try:
#         # 提取 preference 和 score
#         preference = match.group(1)  # 提取偏好响应
#         strength = int(match.group(2))  # 提取偏好强度并转换为整数
#         if preference == "a":
#             prediction = 1
#         elif preference == "b":
#             prediction = 2
#         else:
#             raise Exception("Preference not matched!")
#         if strength not in {1, 2, 3}:
#             raise Exception("Strength not matched!")
#         return {"cot": line, "judgement": prediction, "strength": strength}
#     except:
#         print("Parsing failed! Input is")
#         print(line)
#         return {"cot": line, "judgement": 0, "strength": 0}

def process_output(line, parsing="strict", log_num=1):

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

    global MAX_PARSE_LOG
    if MAX_PARSE_LOG < log_num:
        print("######## Parsing fail! Response is: ########")
        print(line)
        MAX_PARSE_LOG += 1

    return {"cot": line.strip(), "judgement": 0}

@timeout_decorator.timeout(1200)
def request_gpt(line, model, temperature, max_new_tokens):

    api_key = "sk-agEcX3Su78Bu09c2F49978C6Ba424977B936C8710fAb42E0"
    client = openai.OpenAI(api_key=api_key, base_url="https://api.shubiaobiao.cn/v1/")
    payload = {
        "model": model,
        "messages": line['messages'],
    }
    max_tries = 10
    res = ''
    for i in range(max_tries):
        try:
            chat_completion = client.chat.completions.create(model=payload['model'], temperature=temperature, messages=payload['messages'])
            res = chat_completion.choices[0].message.content
        except Exception as e:
            print("Exception! The exception is "+str(e))
            time.sleep(5)
            continue

    judgement = process_output(res)

    if judgement["judgement"] == line['better']:
        if line['better'] == 1:
            chosen_name = "Response (a)"
            rejected_name = "Response (b)"
        else:
            chosen_name = "Response (b)",
            rejected_name = "Response (a)"
        judgement_line = {
                           "system": judge_prompt["system"],
                           "instruction": judge_prompt["user"].format(
                               instruction = line["instruction"],
                               response1 = line["response1"],
                               response2 = line["response2"],
                           ),
                           "input": "",
                           "output": judgement["cot"],
                         }
        critique_line = {
                           "system": critic_prompt["system"],
                           "instruction": critic_prompt["user"].format(
                               instruction = line["instruction"],
                               response1 = line["response1"],
                               response2 = line["response2"],
                               chosen = chosen_name,
                               rejected = rejected_name,
                           ),
                           "input": "",
                           "output": judgement["cot"],
                        }
        with open(args.output_file, "a+", encoding="utf-8") as fjudge:
            fjudge.write(json.dumps(judgement_line)+"\n")
        with open(args.output_critique_file, "a+", encoding="utf-8") as fcritic:
            fcritic.write(json.dumps(critique_line)+"\n")

    counter.value += 1
    if counter.value % 1 == 0:
        avg_time = round((time.time()-start_time) / counter.value, 2)
        print(f"{counter.value} lines finished! {avg_time} seconds per line on average.")

def build_prompt(lines, reverse=False, randomize=False):

    system_prompt = judge_prompt["system"]
    prompt_template = judge_prompt["user"]

    for line in lines:
        prompt = prompt_template.format(instruction=line['instruction'], response1=line['response1'], response2=line['response2'])
        line["messages"] = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
    
    return lines

def infer_judge(lines, model, temperature, max_new_tokens, reverse=False, randomize=False):

    inputs = build_prompt(lines)

    if args.multi_process == "False":
        for line in inputs:
            request_gpt(line, model=model, temperature=temperature, max_new_tokens=max_new_tokens)
    else:
        pool_fn = partial(request_gpt, model=model, temperature=temperature, max_new_tokens=max_new_tokens)
        pool.map(pool_fn, inputs)

def init(c, t):
    global counter
    global start_time
    counter = c
    start_time = t

if __name__ == "__main__":
    parser = build_parser()
    parser.add_argument(
        "--multi-process",
        type=str,
        default="True",
    )
    parser.add_argument(
        "--pool-number",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--output-critique-file",
        type=str,
        required=True,
    )
    args = parser.parse_args()

    manager = multiprocessing.Manager()
    counter = manager.Value("counter", 0)
    start_time = time.time()

    pool = multiprocessing.Pool(processes=args.pool_number, initializer=init, initargs=(counter, start_time))

    fin = open(args.input_file, "r")
    lines = json.load(fin)

    if os.path.exists(args.output_file):
        with open(args.output_file, "r", encoding="utf-8") as fin:
            finished_lines = [json.loads(line) for line in fin.readlines()]
        lines = lines[len(finished_lines):]

    print(f"Totally {len(lines)} lines to infer.")

    infer_judge(lines, model=args.model_path, max_new_tokens=args.max_new_tokens, temperature=args.temperature)