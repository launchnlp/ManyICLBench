from datasets import load_dataset
import collections
import json

from transformers import AutoTokenizer
MATH_DATA_PATH="data/MATH"
class InContextDataset:
    def __init__(self, task, tokenizer):
        # This will return a DatasetDict with keys 'seed0', 'seed1', 'seed2', etc.
        self.dataset = load_dataset("launch/ManyICLBench", task)
        self.dataset_name = task
        self.tokenizer = tokenizer
        self.processed_data = collections.defaultdict(dict)

        seeds = ['seed0', 'seed1', 'seed2']
        lengths = ['1k', '2k', '4k', '8k', '16k', '32k', '64k', '128k']
        for seed in seeds:
            if seed in self.dataset:
                if "MATH" in task:
                    self.processed_data[seed]['max'] = {}
                    seed_dataset = self.dataset[seed]
                    test_examples = seed_dataset['Test Data'][0]
                    test_prompts = []
                    test_targets = []
                    for example in test_examples:
                        with open(MATH_DATA_PATH + f"/test/{task.split('-')[1]}/{example}", "r") as f:
                            data = json.load(f)
                            test_prompts.append(data['problem'])
                            test_targets.append(data['solution'])
                    for length in lengths:
                        max_prompt = ""
                        examples = seed_dataset[length][0]
                        demo_prompt = ""
                        for example in examples:
                            with open(MATH_DATA_PATH + f"/train/{task.split('-')[1]}/{example}", "r") as f:
                                data = json.load(f)
                            demo_prompt += "Problem: " + data['problem'] + "\n"
                            demo_prompt += "Solution: " + data['solution'] + "\n"
                        demo_prompt += "\n"
                        final_prompts = []
                        for i in range(len(test_prompts)):
                            final_prompt = demo_prompt + "Problem: " + test_prompts[i] + "\n" + "Solution: "
                            final_prompts.append(final_prompt)
                            if len(final_prompt + test_targets[i]) > len(max_prompt):
                                max_prompt = final_prompt + test_targets[i]
                        self.processed_data[seed][length] = (final_prompts, test_targets)
                        self.processed_data[seed]['max'][length] = self.tokenizer(max_prompt, return_length=True)['length'][0]
                else:
                    self.processed_data[seed]['max'] = {}
                    seed_dataset = self.dataset[seed]
                    test_prompts = seed_dataset['Test Data'][0]
                    test_targets = seed_dataset['Test Target'][0]

                    for length in lengths:
                        if length in seed_dataset.features:
                            max_prompt = ""
                            in_context_examples = seed_dataset[length][0]
                            
                            final_prompts = []
                            for i in range(len(test_prompts)):
                                final_prompt = in_context_examples + test_prompts[i]
                                final_prompts.append(final_prompt)
                                if len(final_prompt + test_targets[i]) > len(max_prompt):
                                    max_prompt = final_prompt + test_targets[i]
                            self.processed_data[seed][length] = (final_prompts, test_targets)
                            self.processed_data[seed]['max'][length] = self.tokenizer(max_prompt, return_length=True)['length'][0]
    def get_dataset(self, seed, length):
        if seed in self.processed_data and length in self.processed_data[seed]:
            return self.processed_data[seed][length]
        else:
            if seed not in self.dataset:
                raise ValueError(f"Data not found for seed {seed}. Available seeds: {list(self.dataset.keys())}")
            if length not in self.dataset[seed].features:
                 raise ValueError(f"Data not found for length {length} in seed {seed}. Available lengths: {[col for col in self.dataset[seed].features if col != 'Test Examples']}")
            raise ValueError(f"Data not found for seed {seed} and length {length}")
    def get_cache_content(self, seed, length):
        return self.dataset[seed][length][0]
    def get_prompt_texts(self):
        """
        Returns the prompt texts for a given dataset.
        """
        dataset_name = self.dataset_name
        if dataset_name == 'banking77':
            return ['Query: ', 'Intent: ']
        elif dataset_name == 'goEmotions':
            return ['Comment: ', 'Category: ']
        elif dataset_name == 'dialogRE':
            return ['Dialogue: ', 'The respective relations between each entity pair are: ']
        elif dataset_name == 'trec_50':
            return ['Question: ', 'Type: ']
        elif dataset_name == 'clinc150':
            return ['Query: ', 'Intent: ']
        elif dataset_name.startswith('MATH'):
            return ['Problem: ', 'Solution: ']
        elif dataset_name.startswith('BBH'):
            return ['Input: ', 'Target: ']
        elif dataset_name.startswith('ARC'):
            return ['Question: ', 'Answer: ']
        elif dataset_name.startswith('GPQA'):
            return ['Question: ', 'Answer: ']
        elif dataset_name == 'XLSUM':
            return ['Article: ', 'Summary: ']
        elif dataset_name.startswith('MT'):
            language = dataset_name[3:]
            return ['English: ', f"{language}: "]
        elif dataset_name == 'GSM8K':
            return ['Question: ', 'Answer: ']
        else:
            raise NotImplementedError(f"Prompt texts not defined for dataset: {dataset_name}")
if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", use_fast=True, trust_remote_code=True)
    ds = InContextDataset("ARC-Challenge", tokenizer)
    prompts, targets = ds.get_dataset('seed0', '1k')
    print(f"Number of prompts for seed0, 1k: {len(prompts)}")
    print(f"Number of targets for seed0, 1k: {len(targets)}")
    print(f"Max length for seed0, 1k: {ds.processed_data['seed0']['max']}")
    if len(prompts) > 0:
        print("\nExample prompt:")
        print(prompts[0])
        print("\nCorresponding target:")
        print(targets[0])
    