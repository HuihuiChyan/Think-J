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

    score_pattern = r"Therefore, Response \((a|b)\) is better, and the strength is \[\[(\d+)\]\]\."
    
    # 使用正则表达式匹配输入字符串
    match = re.search(score_pattern, solution_str)

    if match is None:
        reward = 0.0

    else:
        # 提取 preference 和 score
        preference = match.group(1)  # 提取偏好响应
        strength = int(match.group(2))  # 提取偏好强度并转换为整数

        if (preference == "a" and ground_truth < 0) or (preference == "b" and ground_truth > 0):
            outcome_reward = 1

            if strength not in {1, 2, 3}:
                strength_reward = - 0.5
            else:
                strength_reward = - abs(abs(ground_truth) - strength) / 8

            reward = outcome_reward + strength_reward

        else:
            reward = 0.0

    # if random.random() > 0.9:
    #     print(f"{reward}\n{solution_str}")
    
    return reward