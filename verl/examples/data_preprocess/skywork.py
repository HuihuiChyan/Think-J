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
"""
Preprocess the GSM8k dataset to parquet format
"""

import re
import os
import json
import datasets

from verl.utils.hdfs_io import copy, makedirs
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='../../data/skywork')
    parser.add_argument('--hdfs_dir', default=None)

    args = parser.parse_args()
    
    train_dataset = datasets.load_dataset('json', data_files='/home/hh456524/WorkSpace/STaR-Judge-data/skywork/skywork-infer-R1-critique-train.json')
    test_dataset = datasets.load_dataset('json', data_files='/home/hh456524/WorkSpace/STaR-Judge-data/skywork/skywork-infer-R1-critique-test.json')

    # add a row to each data item that represents a unique id
    def make_map_fn(split):

        def process_fn(example, idx):
            system = example.pop('system')
            question = example.pop('instruction')
            answer = example.pop('output')
            input = example.pop('input')

            # output_pattern = {
            #     1: r"Response \(a\) is better.$",
            #     2: r"Response \(b\) is better.$"
            # }
            if re.search(r"Response \(a\) is better.$", answer):
                solution = 1
            else:
                solution = 2

            data = {
                "data_source": "skywork/skywork-reward-preference-80k-v0.2",
                "prompt": [
                    {
                        "role": "system",
                        "content": system,
                    },
                    {
                        "role": "user",
                        "content": question,
                    },                    
                ],
                "ability": "judge",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": solution
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                }
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('train'), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset['train'].to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset['train'].to_parquet(os.path.join(local_dir, 'test.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
