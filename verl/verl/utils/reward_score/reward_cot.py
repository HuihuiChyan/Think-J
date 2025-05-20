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

    output_pattern = {
        1: r"Response \(a\) is better.",
        2: r"Response \(b\) is better."
    }
    judgement = 0
    for prediction, pattern in output_pattern.items():
        if re.search(pattern, solution_str):
            judgement = prediction
    if judgement == ground_truth:
        return 1.0
    elif judgement == 0:
        # Parsing failed
        return 0.5
    else:
        # Wrong judgement
        return 0.0