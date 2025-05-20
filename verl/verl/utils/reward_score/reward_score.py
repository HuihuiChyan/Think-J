# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import random

def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    """The scoring function for GSM8k.

    Reference: Trung, Luong, et al. "Reft: Reasoning with reinforced fine-tuning." Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2024.

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """

    score_pattern = r"<Score 1>(\d*\.?\d+)</Score 1>.*<Score 2>(\d*\.?\d+)</Score 2>"
    match = re.search(score_pattern, solution_str)

    if match:
        # 提取分数并转换为浮点数
        score1 = float(match.group(1))
        score2 = float(match.group(2))

        if (score1 > score2 and ground_truth == 1) or (score1 < score2 and ground_truth == 2):
            result = min(abs(score1 - score2)/100, 1.0) + 1.0
        else:
            result = max(-abs(score1 - score2)/100, -1.0)
    else:
        result = 0.5
    
    # if random.random() > 0.9:
    #     print(f"{result}\n{solution_str}")

    return result