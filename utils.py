import re
import nltk
from rouge_score import rouge_scorer
import sacrebleu
from sacrebleu import CHRF
from sklearn.metrics import f1_score, accuracy_score
import os
import json
scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

def evaluate(task, preds, targets,):
    preds = [parse_output(task, p) for p in preds]
    targets = [parse_output(task, t) for t in targets]
    if task == 'XLSUM':
        result = 0
        for pred, target in zip(preds, targets):
            pred = add_newline_to_end_of_each_sentence(pred)
            target = add_newline_to_end_of_each_sentence(target)
            scores = scorer.score(pred, target)
            result += scores['rougeL'].fmeasure
        return result / len(preds)
    elif task[:2] == 'MT':
        chrf = CHRF(word_order=2)
        targets = [[t] for t in targets]
        return chrf.corpus_score(preds, targets).score
    elif 'banking77' in task or 'goEmotions' in task or 'trec_50' in task or 'clinc150' in task:
        return f1_score(targets, preds, average='macro')
    elif 'dialogRE' in task:
        new_preds = []
        for t, p in zip(targets, preds):
            new_preds.extend(p[:len(t)])
            for i in range(len(t) - len(p)):
                new_preds.append('unknown')
        targets = [t for target in targets for t in target]
        return f1_score(targets, new_preds, average='macro')
    elif "MATH" in task:
        correct = 0
        for pred, target in zip(preds, targets):
            if pred == target:
                correct += 1
        return correct / len(preds)
    else:
        return accuracy_score(targets, preds)

def parse_output(task, output):
    output = output.strip('\n')
    if 'MATH' in task:
        result = last_boxed_only_string(output)
        if result:
            return result
        else:
            return ""
    elif task == 'dialogRE':
         output = output.split('\n')[0].lower().split(',')
         output = [o.strip() for o in output]
         return output
    elif task == "GSM8K":
        pattern = r'#### (\d+)'
        match = re.search(pattern, output)
        if match:
            return match.group(1)
        else:
            return ""
    else:
        label = output.split('\n')[0].strip()
        return label
    
def add_newline_to_end_of_each_sentence(x: str) -> str:
    """This was added to get rougeLsum scores matching published rougeL scores for BART and PEGASUS."""
    re.sub("<n>", "", x)  # remove pegasus newline char
    return "\n".join(nltk.sent_tokenize(x))

def create_batched_inputs(inputs, batch_size):
    num_batches = len(inputs) // batch_size
    if len(inputs) % batch_size != 0:
        num_batches += 1
    batched_inputs = []
    for i in range(num_batches):
        batched_inputs.append(inputs[i * batch_size:(i + 1) * batch_size])
    return batched_inputs
def remove_prefix(text, outputs, return_tensors='pt', add_special_tokens=False):
    '''This method remove prefix in llama.
    '''
    spaces = {'\n', '\t', ' '}
    if isinstance(text, str):
        if text[0] in spaces and not add_special_tokens:
            if return_tensors is None:
                outputs.input_ids = outputs.input_ids[1:]
                outputs.attention_mask = outputs.attention_mask[1:]
            elif return_tensors == 'pt':
                outputs.input_ids = outputs.input_ids[:, 1:]
                outputs.attention_mask = outputs.attention_mask[:, 1:]
                
    elif isinstance(text, list):
        if text[0][0] in spaces and not add_special_tokens: # assume all the input are started with spaces
            if return_tensors is None:
                outputs.input_ids = list(map(lambda x: x[1:], outputs.input_ids))
                outputs.attention_mask = list(map(lambda x: x[1:], outputs.attention_mask))
            elif return_tensors == 'pt':
                outputs['input_ids'] = outputs.input_ids[:, 1:]
                outputs['attention_mask'] = outputs.attention_mask[:, 1:]
    return outputs

def choose_batch_size(model_name, context_length):
    bs_mapping = json.load(open('bs_mapping.json'))
    return bs_mapping[model_name][str(context_length)]

def last_boxed_only(sample):
    """
    Given a (q,a) sample, filter the answers so that they only contain 
    the last \boxed{...} or \fbox{...} element
    """
    q, a = sample
    a = last_boxed_only_string(a)
    if a == None:
        return None
    return (q, a)

def last_boxed_only_string(string):
    idx = string.find("\\boxed")
    if idx < 0:
        idx = string.find("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1
    
    if right_brace_idx == None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]
    
    return retval