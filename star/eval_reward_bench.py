import os
import json
import torch
import argparse
from utils_args import build_parser
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from infer_judge import infer_judge

def make_data_row(id: int, instruction: str, response1: str, response2: str, label: int) -> dict:
    return {
        "instruction": instruction.strip(),
        "response1": response1.strip(),
        "response2": response2.strip(),
        "better": 1,
    }

def load_rewardbench(data_path, debug=False):
    SUBSET_MAPPING = {
        "Chat": [
            "alpacaeval-easy",
            "alpacaeval-length",
            "alpacaeval-hard",
            "mt-bench-easy",
            "mt-bench-med",
        ],
        "Chat Hard": [
            "mt-bench-hard",
            "llmbar-natural",
            "llmbar-adver-neighbor",
            "llmbar-adver-GPTInst",
            "llmbar-adver-GPTOut",
            "llmbar-adver-manual",
        ],
        "Safety": [
            "refusals-dangerous",
            "refusals-offensive",
            "xstest-should-refuse",
            "xstest-should-respond",
            "donotanswer",
        ],
        "Math": [
            "math-prm",
        ],
        "Code": [
            "hep-cpp",
            "hep-go",
            "hep-java",
            "hep-js",
            "hep-python",
            "hep-rust",
        ]
    }
    dataset = []
    with open(data_path) as fin:
        lines = [line.strip() for line in fin.readlines()]
        dataset = [json.loads(line) for line in lines]
    
    benchmark_set = {}
    for subset_name in ["Chat", "Chat Hard", "Safety", "Math", "Code"]:
        subset = []
        for i, row in enumerate(dataset):
            if dataset[i]["subset"] in SUBSET_MAPPING[subset_name]:
                subset.append(make_data_row(i, row["prompt"], row["chosen"], row["rejected"], 1))
        if debug:
            benchmark_set[subset_name] = subset[:50]
        else:
            benchmark_set[subset_name] = subset
            
    return benchmark_set


def cal_statistics(all_judgements, all_judgements_rev):

    all_stats = {}

    for subset_name in all_judgements.keys():

        stats = {key: 0 for key in ["single_total", "single_correct", "single_accuracy", "parsing_failure",
                                    "pair_total", "pair_correct", "pair_accuracy", "pair_agree", "pair_agreement_rate"]}
        
        judgements = all_judgements[subset_name]
        judgements_rev = all_judgements_rev[subset_name]
        for i in range(len(judgements)):
            stats["single_total"] += 2
            stats["pair_total"] += 1
            if judgements[i]["judgement"] == 1:
                stats["single_correct"] += 1
            if judgements_rev[i]["judgement"] == 2:
                stats["single_correct"] += 1
            if judgements[i]["judgement"] == 1 and judgements_rev[i]["judgement"] == 2:
                stats["pair_correct"] += 1
            if set([judgements[i]["judgement"], judgements_rev[i]["judgement"]]) in [set([1, 2]), set([0])]:
                stats["pair_agree"] += 1
            if judgements[i]["judgement"] == 0:
                stats["parsing_failure"] += 1
            if judgements_rev[i]["judgement"] == 0:
                stats["parsing_failure"] += 1

            stats["single_accuracy"] = round(
                stats["single_correct"] / stats["single_total"]*100, 2)
            stats["pair_accuracy"] = round(
                stats["pair_correct"] / stats["pair_total"]*100, 2)
            stats["pair_agreement_rate"] = round(
                stats["pair_agree"] / stats["pair_total"]*100, 2)
            stats["failure_rate"] = round(
                stats["parsing_failure"] / stats["single_total"]*100, 2)

        del stats["single_total"]
        del stats["single_correct"]
        del stats["pair_total"]
        del stats["pair_correct"]
        del stats["pair_agree"]
        del stats["parsing_failure"]

        all_stats[subset_name] = stats

    all_stats["Reasoning"] = {}
    for metric in ["single_accuracy", "pair_accuracy", "pair_agreement_rate", "failure_rate"]:
        all_stats["Reasoning"][metric] = round((all_stats["Math"][metric] + all_stats["Code"][metric]) / 2, 2)

    del all_stats["Math"]
    del all_stats["Code"]

    all_stats["Average"] = {}
    for metric in ["single_accuracy", "pair_accuracy", "pair_agreement_rate", "failure_rate"]:
        all_stats["Average"][metric] = round((all_stats["Chat"][metric] + all_stats["Chat Hard"][metric] + \
                                       all_stats["Safety"][metric] + all_stats["Reasoning"][metric]) / 4, 2)

    return all_stats


if __name__ == "__main__":
    parser = build_parser()
    parser.add_argument("--debug", choices=("True", "False"), default="False")
    args = parser.parse_args()

    data = load_rewardbench("data/rewardbench/filtered.json", debug=(args.debug=="True"))

    model = LLM(model=args.model_path, tensor_parallel_size=torch.cuda.device_count(), trust_remote_code=True, max_model_len=8192)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    all_stats = {}
    all_judgements = {}
    all_judgements_rev = {}

    for subset_name in data.keys():
        all_judgements[subset_name], _  = infer_judge(data[subset_name], model=model, tokenizer=tokenizer, max_new_tokens=args.max_new_tokens, \
                                                      parsing="loose", temperature=args.temperature, prompt_type=args.prompt_type)
        all_judgements_rev[subset_name], _  = infer_judge(data[subset_name], model=model, tokenizer=tokenizer, max_new_tokens=args.max_new_tokens, \
                                                          parsing="loose", temperature=args.temperature, reverse=True, prompt_type=args.prompt_type)

        for i in range(len(all_judgements[subset_name])):
            all_judgements[subset_name][i]["data"] = data[subset_name][i]
            all_judgements_rev[subset_name][i]["data"] = data[subset_name][i]

    all_stats = cal_statistics(all_judgements, all_judgements_rev)

    print(f"Evaluating {args.model_path} with prompt type as {args.prompt_type}:")
    for subset_name in all_stats.keys():
        print(f"Results on {subset_name}: {all_stats[subset_name]}")
    
    model_name = args.model_path.split("/")[-1]
    if model_name == "huggingface":
        model_name = args.model_path.split("/")[-4]
    with open(f"./results/{model_name}_result.json", "w") as fout:
        json.dump(all_stats, fout, indent=4)
    with open(f"./results/{model_name}_judgements.json", "w") as fout:
        judgements = {"forward": all_judgements, "backward": all_judgements_rev}
        json.dump(judgements, fout, indent=4)