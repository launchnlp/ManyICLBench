# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
Prepare prediction jsonl with field `pred` .
dataset jsonl:
{
    "index" int,
    "input": str,
    "outputs": [str],
}

prediction jsonl: 
{
    "index" int,
    "input": str,
    "outputs": [str],
    "pred": str,
}
"""

import argparse
import json
import os
import sys
import threading
import time
from tqdm import tqdm
from pathlib import Path
import traceback
import logging
import numpy as np
import pickle
import csv

from transformers import AutoTokenizer
from datetime import date
from dataset import InContextDataset
from utils import *

model_mapping = {
    'Llama-3.1-8B': 'meta-llama/Meta-Llama-3.1-8B',
    'gemma-2-9b': 'google/gemma-2-9b',
    'Mistral-Nemo-Base': 'mistralai/Mistral-Nemo-Base-2407',
    'glm-4-9b-chat': 'THUDM/glm-4-9b-chat',
    'glm-4-9b': 'THUDM/glm-4-9b',
    'Llama-3.1-8B-Instruct': 'meta-llama/Meta-Llama-3.1-8B-Instruct',
    'Llama-3.1-70B-Instruct': 'meta-llama/Meta-Llama-3.1-70B-Instruct',
    'Mistral-Nemo-Instruct': 'mistralai/Mistral-Nemo-Instruct-2407',
    'Mistral-Large-Instruct': 'mistralai/Mistral-Large-Instruct-2407',
    'Mistral-Large-Instruct-GGUF': "bartowski/Mistral-Large-Instruct-2407-GGUF",
    'gemma-2-9b-it': 'google/gemma-2-9b-it',
    'command-r':'CohereForAI/c4ai-command-r-v01',
    'Meta-Llama-3.1-405B-Instruct-4bit': 'hugging-quants/Meta-Llama-3.1-405B-Instruct-GPTQ-INT4',
    'Qwen2-7B-Instruct':'Qwen/Qwen2-7B-Instruct',
    'Qwen2-72B-Instruct':'Qwen/Qwen2-72B-Instruct',
    'Qwen2-7B-Instruct-AWQ':'Qwen/Qwen2-7B-Instruct-AWQ',
    'Qwen2-72B-Instruct-AWQ':'Qwen/Qwen2-72B-Instruct-AWQ',
    'Llama-3.1-70B-Instruct-AWQ':"hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4",
    'Llama-3.1-8B-Instruct-AWQ':"hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
    "Mistral-Large-Instruct-AWQ": "casperhansen/mistral-large-instruct-2407-awq",
    "Mistral-Nemo-Instruct-AWQ": "casperhansen/mistral-nemo-instruct-2407-awq",
    'Jamba-1.5-Mini': 'ai21labs/AI21-Jamba-1.5-Mini',
    'Phi-3-Mini-Instruct': 'microsoft/Phi-3-mini-128k-instruct',
    'Phi-3-Small-Instruct': 'microsoft/Phi-3-small-128k-instruct',
    'Phi-3-Medium-Instruct': 'microsoft/Phi-3-medium-128k-instruct'
}

logger = logging.getLogger(__name__)

def set_file_handler(logger, file_name):
    today = date.today().strftime("%b-%d-%Y")
    log_dir = os.path.join("./logs", today)
    os.makedirs(log_dir, exist_ok=True)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_path = os.path.join(log_dir, file_name)
            
    fh = logging.FileHandler(filename=log_path, mode='a')
    formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

SERVER_TYPES = (
    'vllm',
    'openai',
    'gemini'
)


class ServerAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        namespace.server_type = values


parser = argparse.ArgumentParser()
# Data
parser.add_argument('--dataset', type=str, required=True, help='dataset', dest='dataset', action='store')
parser.add_argument('--mode', type=str, choices=['full', 'remove_closest', 'remove_furthest', 'only_closest', 'corrupt',"replace_closest", 'corrupt_closest', 'corrupt_furthest', 'sort_front_closest', 'sort_middle_closest', 'sort_end_closest'], help='mode', dest='mode', action='store', default='full')
parser.add_argument('--seeds', type=int, required=True, help='number of seeds to try', dest='seeds', action='store')
parser.add_argument('--context_length', type=int, help='context length', dest='context_length', action='store')
parser.add_argument('--cot', action='store_true', help='use cot', default=False)
parser.add_argument('--recompute', action='store_true', help='recompute', default=False)
parser.add_argument('--self_tokenizer', action='store_true', help='use self tokenizer', default=False)
parser.add_argument('--duplicate_length', type=int, help='duplicate length', dest='duplicate_length', action='store', default=200000)
parser.add_argument('--shot', type=int, help='shot', dest='shot', action='store', default=0)
parser.add_argument('--retriever', type=str, help='retriever', dest='retriever', action='store', default='BM25')
# Server
parser.add_argument("--server_type", default='nemo', action=ServerAction, choices=SERVER_TYPES)
parser.add_argument("--server_host", type=str, default='127.0.0.1')
parser.add_argument("--server_port", type=str, default='5000')
parser.add_argument("--ssh_server", type=str)
parser.add_argument("--ssh_key_path", type=str)
parser.add_argument("--model_name_or_path", type=str, default='gpt-3.5-turbo', 
                    help='supported models from OpenAI or HF (provide a key or a local path to the checkpoint)')

# Inference
parser.add_argument("--temperature", type=float, default=0)
parser.add_argument("--top_k", type=int, default=32)
parser.add_argument("--top_p", type=float, default=1.0)
parser.add_argument("--random_seed", type=int, default=0)
parser.add_argument("--stop_words", type=str, default='')
parser.add_argument("--sliding_window_size", type=int)
parser.add_argument("--threads", type=int, default=4)
parser.add_argument("--batch_size", type=int, default=1)

args = parser.parse_args()
args.stop_words = list(filter(None, args.stop_words.split(',')))
if args.server_type == 'hf' or args.server_type == 'gemini':
    args.threads = 1


def get_llm(tokens_to_generate, cache_content=None):
    if args.server_type == 'vllm':
        from client_wrappers import VLLMClient
        llm = VLLMClient(
            server_host=args.server_host,
            server_port=args.server_port,
            ssh_server=args.ssh_server,
            ssh_key_path=args.ssh_key_path,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            random_seed=args.random_seed,
            stop=args.stop_words,
            tokens_to_generate=tokens_to_generate,
        )
 
    elif args.server_type == 'openai':
        from client_wrappers import OpenAIClient
        llm = OpenAIClient(
            model_name=args.model_name_or_path,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            random_seed=args.random_seed,
            stop=args.stop_words,
            tokens_to_generate=tokens_to_generate,
        )

    elif args.server_type == 'gemini':
        from client_wrappers import GeminiClient
        llm = GeminiClient(
            model_name=args.model_name_or_path,
            cache_content=cache_content,
            project_id=args.project_id,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            random_seed=args.random_seed,
            stop=args.stop_words,
            tokens_to_generate=tokens_to_generate,
        )
    else:
        raise RuntimeError(f'Unsupported server type {args.server_type}')

    return llm


def main():
    start_time = time.time()
    
    params = {
        'dataset': args.dataset,
        'mode': args.mode,
        'model_name': args.model_name_or_path,
        'context_length': args.context_length,
    }
    expr_name = f"{params['model_name']}_{params['dataset']}_{params['mode']}"
    log_file_name = f"{expr_name}_{args.seeds}.log"
    set_file_handler(logger, log_file_name)
    params['logger'] = logger
    logger.info(f"Starting experiment: {expr_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_mapping[params['model_name']], use_fast=True, token=access_token, trust_remote_code=True)

    dataset = InContextDataset(params['dataset'], tokenizer)

    
        
    for length in ['1k', '2k', '4k', '8k', '16k', '32k', '64k', '128k']:
        scores = []
        for seed in range(args.seeds):
            params['seed'] = seed
            curr_expr_name = f"{expr_name}_{length}_{seed}"
            logger.info(f"Starting seed: {seed}")
            #get prompts and targets
            prompts, targets = dataset.get_dataset("seed"+str(seed), length)
            logger.info(f"Finish loading prompts and targets")

            data = []
            for idx, prompt, target in zip(range(len(prompts)), prompts, targets):
                data.append({
                    'index': idx,
                    'input': prompt,
                    'outputs': target,
                })
            if not os.path.exists(f"preds/{params['dataset']}"):
                os.makedirs(f"preds/{params['dataset']}")
            pred_file = f"preds/{params['dataset']}/{curr_expr_name}_prompts.jsonl"
            #empty pred_file
            if os.path.exists(pred_file) and not args.recompute:
                with open(pred_file, 'r'):
                    outputs_parallel = [json.loads(line) for line in open(pred_file)]
                    existing_pred = {o['index']: o for o in outputs_parallel}
                    #filter out the ones that are not in the test data and output is empty
                    existing_pred = {k:v for k, v in existing_pred.items() if v['pred'] != ''}
                    outputs_parallel = []
                    data = [d for d in data if d['index'] not in existing_pred]
                    logger.info(f"Existing predictions: {len(existing_pred)}")
            else:
                existing_pred = {}
                outputs_parallel = []
            if len(dataset.get_prompt_texts()[0]) > 0:
                args.stop_words = [f"\n{dataset.get_prompt_texts()[0]}"]
            if args.server_type == 'gemini' and len(data) != 0:
                cache_content = dataset.get_cache_content("seed"+str(seed), length)
                if len(tokenizer.encode(cache_content)) > 32769:
                    for d in data:
                        d['input'] = d['input'][len(cache_content):]
            else:
                cache_content = None
            # Load api
            llm = get_llm(dataset.processed_data['seed0']['max'][length] + 5, cache_content=cache_content)
            args.batch_size = choose_batch_size(params['model_name'], params['context_length'])
            if length == "128k":
                args.threads = 2
            def get_output(idx_list, index_list, input_list, outputs_list, others_list, truncation_list, length_list):
                nonlocal llm
                count = 0
                while True:
                    try:
                        pred_list = llm.process_batch(prompts=input_list)
                        break
                    except Exception as e:
                        traceback.print_exc()
                        count += 1
                        if count > 3:
                            raise e

                zipped_iter = zip(pred_list, idx_list, index_list, input_list,
                                outputs_list, others_list, truncation_list, length_list)

                for pred, idx, index, input, outputs, others, truncation, length in zipped_iter:
                    if isinstance(pred['text'], str):
                        pred_text = pred['text']
                    elif len(pred['text']) > 0:
                        pred_text = pred['text'][0]
                    else:
                        pred_text = ''

                    outputs_parallel[idx] = {
                        'index': index,
                        'pred': pred_text,
                        'outputs': outputs,
                        'others': others,
                        'truncation': truncation,
                        'length': length,
                    }
            
            if len(data) != 0:
                with open(pred_file, 'w') as f:
                    pass
                threads = []
                outputs_parallel = [{} for _ in range(len(data))]

                batched_data = []
                batch = []
                for idx, data_point in enumerate(data):
                    data_point['idx'] = idx

                    if len(batch) >= args.batch_size:
                        batched_data.append(batch)
                        batch = []

                    batch.append(data_point)

                if len(batch):
                    batched_data.append(batch)

                # setting buffering=1 to force to dump the output after every line, so that we can see intermediate generations
                with open(pred_file, 'at', encoding="utf-8", buffering=1) as fout:
                    # the data is processed sequentially, so we can store the start and end of current processing window
                    start_idx = 0  # window: [start_idx, end_idx]

                    for batch_idx, batch in tqdm(enumerate(batched_data), total=len(batched_data)):
                        idx_list = [data_point['idx'] for data_point in batch]
                        end_idx = idx_list[-1]  # the data in a batch is ordered

                        thread = threading.Thread(
                            target=get_output,
                            kwargs=dict(
                                idx_list=idx_list,
                                index_list=[data_point['index'] for data_point in batch],
                                input_list=[data_point['input'] for data_point in batch],
                                outputs_list=[data_point['outputs'] for data_point in batch],
                                others_list=[data_point.get('others', {}) for data_point in batch],
                                truncation_list=[data_point.get('truncation', -1) for data_point in batch],
                                length_list=[data_point.get('length', -1) for data_point in batch],
                            ),
                        )
                        thread.start()
                        threads.append(thread)

                        is_last_batch = (batch_idx == len(batched_data) - 1)

                        if (len(threads) == args.threads) or is_last_batch:
                            for thread in threads:
                                thread.join()
                            threads = []

                            # dump the results in current processing window on disk
                            for idx in range(start_idx, end_idx + 1):
                                if len(outputs_parallel[idx]) > 0:
                                    fout.write(json.dumps(outputs_parallel[idx]) + '\n')

                            start_idx = end_idx + 1
            if len(data) != 0:
                with open(pred_file, 'at', encoding="utf-8", buffering=1) as fout:
                    for output in existing_pred.values():
                        fout.write(json.dumps(output) + '\n')
            outputs_parallel.extend(existing_pred.values())
            predictions = [o['pred'] for o in outputs_parallel]
            targets = [o['outputs'] for o in outputs_parallel]
            score = evaluate(args.dataset, predictions, targets)
            scores.append(score)

            logger.info(f"Score: {score}")
    
        logger.info(f"Avg Across Seeds: {np.mean(scores)}")
        logger.info(f"Std Across Seeds: {np.std(scores)}")

        results = {}
        results['scores'] = scores
        results['avg_score'] = np.mean(scores)
        results['std_score'] = np.std(scores)
        # Write results to CSV
        os.makedirs(f"./results/csv/{args.dataset}", exist_ok=True)
        with open(f"./results/csv/{args.dataset}/{expr_name}_{length}.csv", mode='w', newline='') as file:
            writer = csv.writer(file)
            # Write header
            writer.writerow(results.keys())
            # Write data
            writer.writerow(results.values())
        if length == f"{int(params['context_length']/1000)}k":
            break

    print(f"Used time: {round((time.time() - start_time) / 60, 1)} minutes")


if __name__ == '__main__':
    main()