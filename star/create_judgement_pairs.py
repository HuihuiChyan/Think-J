import sys
import json
import random
from utils_prompts import direct_judge_prompt as judge_prompt

file_jud = sys.argv[1]
file_pos = sys.argv[2]
file_neg = sys.argv[3]
file_dpo = sys.argv[4]
file_sft = sys.argv[5]

with open(file_jud, "r") as fjud, open(file_pos, "r") as fpos,\
open(file_neg, "r") as fneg, open(file_dpo, "w") as fdpo, open(file_sft, "w") as fsft:
    lines_jud = json.load(fjud)
    lines_pos = json.load(fpos)
    lines_neg = json.load(fneg)

    dpo_lines = []
    sft_lines = []
    JUD_FAIL_CNT = 0
    POS_FAIL_CNT = 0
    NEG_FAIL_CNT = 0
    from utils_prompts import strength_judge_prompt
    for i in range(len(lines_jud)):
        if lines_jud[i]["judgement"] == lines_jud[i]["better"]:
            if lines_neg[i]["judgement"] != lines_jud[i]["better"]:
                if lines_jud[i]["cot"] == "" or lines_neg[i]["cot"] == "":
                    continue
                dpo_lines.append(
                    {
                        "system": judge_prompt["system"],
                        "instruction": judge_prompt["user"].format(
                            instruction = lines_jud[i]["instruction"],
                            response1 = lines_jud[i]["response1"],
                            response2 = lines_jud[i]["response2"],
                        ),
                        "chosen": lines_jud[i]["cot"],
                        "rejected": lines_neg[i]["cot"],
                    }
                )
                sft_lines.append(
                    {
                        "system": judge_prompt["system"],
                        "instruction": judge_prompt["user"].format(
                            instruction = lines_jud[i]["instruction"],
                            response1 = lines_jud[i]["response1"],
                            response2 = lines_jud[i]["response2"],
                        ),
                        "input": "",
                        "output": lines_jud[i]["cot"],
                    }
                )
            else:
                NEG_FAIL_CNT += 1
                continue

        else:
            JUD_FAIL_CNT += 1
            if lines_pos[i]["judgement"] == lines_jud[i]["better"]:
                if lines_jud[i]["cot"] == "" or lines_pos[i]["cot"] == "":
                    continue
                dpo_lines.append(
                    {
                        "system": judge_prompt["system"],
                        "instruction": judge_prompt["user"].format(
                            instruction = lines_jud[i]["instruction"],
                            response1 = lines_jud[i]["response1"],
                            response2 = lines_jud[i]["response2"],
                        ),
                        "chosen": lines_pos[i]["cot"],
                        "rejected": lines_jud[i]["cot"],
                    }
                )
                sft_lines.append(
                    {
                        "system": judge_prompt["system"],
                        "instruction": judge_prompt["user"].format(
                            instruction = lines_jud[i]["instruction"],
                            response1 = lines_jud[i]["response1"],
                            response2 = lines_jud[i]["response2"],
                        ),
                        "input": "",
                        "output": lines_pos[i]["cot"],
                    }
                )
            else:
                POS_FAIL_CNT += 1
                continue       
    
    sample_lines = random.sample(dpo_lines, k=3)
    for i in range(len(sample_lines)):
        sample_line = sample_lines[i]
        print(f"*************Sampled Line {i}*************")
        print(json.dumps(sample_line, indent=4))
        
    print("Totally "+str(len(lines_jud))+" lines before processing.")
    print("Totally "+str(len(dpo_lines))+" lines after processing.")

    print(f"{JUD_FAIL_CNT} lines failed to generate correct judgement.")
    print(f"{POS_FAIL_CNT} lines failed to generate positive critique.")
    print(f"{NEG_FAIL_CNT} lines failed to generate negative critique.")

with open(file_dpo, "w") as fout:
    json.dump(dpo_lines, fout, indent=4)

with open(file_sft, "w") as fout:
    json.dump(sft_lines, fout, indent=4)