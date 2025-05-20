import json
import copy
import re

# Substitute
# Given that ('Response (b)',) is better than Response (a)
# Given that Response (b) is better than Response (a)

INPUT = "./skywork707-cot-671BR1-judgement.jsonl"
OUTPUT1 = "./skywork707-cot-671BR1-judgement.json"
OUTPUT2 = "./skywork707-cot-671BR1-critique.json"

with open(INPUT, "r", encoding="utf-8") as fin,\
open(OUTPUT1, "w", encoding="utf-8") as fout1,\
open(OUTPUT2, "w", encoding="utf-8") as fout2:

    lines = [json.loads(line.strip()) for line in fin.readlines()]
    newlines = []
    critique_lines = []
    for line in lines:

        # 正则表达式
        pattern = r"<think>.*?</think>"

        # 使用 re.findall 查找所有匹配的 <think> 标签
        matches = re.findall(pattern, line['output'], flags=re.DOTALL)

        # 去掉<think>标签及其内容
        text_without_think = re.sub(pattern, "", line['output'], flags=re.DOTALL).strip()

        # pattern = r'^(.*?)(Therefore, Response \((a|b)\) is better, and the strength is \[\[(1|2|3)\]\].)'
        pattern = r"^(.*?)(Therefore, Response \((a|b)\) is better\.)$"
        match = re.search(pattern, text_without_think, re.DOTALL)
        
        assert match is not None

        before_therefore = match.group(1).strip()

        if before_therefore == "":
            continue

        there = match.group(2).strip()
        # 在两边加上 <Think> 标签
        line['output'] = f'<think>\n{before_therefore}\n</think>\n\n{there}'

        critique_line = copy.deepcopy(line)

        if re.match(r"Therefore, Response \(a\) is better", there):
            critique_line['instruction'] = critique_line['instruction'].replace(
                "Please provide an evaluation by first offering a detailed explanation.",
                "Given that Response (a) is better than Response (b), please provide an evaluation by first offering a detailed explanation.",)
        elif re.match(r"Therefore, Response \(b\) is better", there):
            critique_line['instruction'] = critique_line['instruction'].replace(
                "Please provide an evaluation by first offering a detailed explanation.",
                "Given that Response (b) is better than Response (a), please provide an evaluation by first offering a detailed explanation.",)
        else:
            raise Exception("Critique error!")

        newlines.append(line)
        critique_lines.append(critique_line)

    json.dump(newlines, fout1, indent=4, ensure_ascii=False)
    json.dump(critique_lines, fout2, indent=4, ensure_ascii=False)