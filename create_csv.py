import json
import pandas as pd
import os
import numpy as np

OUTPUT_PATH = './results/aggregated'
tasks = ['banking77', 'goEmotions', 'dialogRE', 'trec_50', 'clinc150', 'MATH-algebra','MATH-geometry', 'MATH-counting_and_probability', 
'MATH-number_theory', 'BBH-geometric_shapes', 'BBH-salient_translation_error_detection', 'GPQA', 'ARC-Easy', 'ARC-Challenge', 'XLSUM', 
'MT_Kurdish', 'MT_Tamil', 'MT_Chinese', "MT_Spanish", 'GSM8K', 'BBH-word_sorting', 'BBH-dyck_languages', "GPQA_cot"]
SSL_tasks = ['banking77', 'dialogRE', 'trec_50', 'clinc150','BBH-geometric_shapes']
ASL_tasks = ['GSM8K', 'MATH-algebra', 'MATH-geometry', 'MATH-counting_and_probability', 'XLSUM', 'BBH-salient_translation_error_detection', 'GPQA_cot','BBH-dyck_languages','MATH-number_theory','ARC-Challenge','BBH-word_sorting']
classification_tasks = ['banking77', 'goEmotions', 'dialogRE', 'trec_50', 'clinc150']
translation_tasks = ['MT_Kurdish', 'MT_Chinese', 'MT_Spanish']
summarization_tasks = ['XLSUM']
math_tasks = ['MATH-algebra', 'MATH-geometry', 'MATH-counting_and_probability', 'MATH-number_theory', 'GSM8K']
science_tasks = ['GPQA', 'GPQA_cot', 'ARC-Easy', 'ARC-Challenge']
symbolic_tasks = ['BBH-geometric_shapes', 'BBH-salient_translation_error_detection','BBH-word_sorting', 'BBH-dyck_languages']


def get_task_result(dataset, model_name, measurement, context_length):
    expr_name = f"{model_name}_{dataset}_{measurement}_{context_length}"
    if os.path.exists(f"./results/csv/{dataset}/{expr_name}.csv"):
        return pd.read_csv(f"./results/csv/{dataset}/{expr_name}.csv")
    else:
        return None
def create_csv_file(datasets, model_name, measurement, context_lengths=None):
    results = {}
    for dataset in datasets:
        for context_length in context_lengths:
            result = get_task_result(dataset, model_name, measurement, context_length)
            temp = dataset
            if temp not in results:
                results[temp] = {}
            if result is not None:
                results[temp][context_length] = result['avg_score'].iloc[0]
            else:
                results[temp][context_length] = -1
    df = pd.DataFrame(results).T
    df.index.name = 'Task'
    task_no_apply = ['MT_Kurdish', 'MT_Tamil', 'MT_Chinese', 'MT_Spanish']
    for task in df.index:
        if task not in task_no_apply:
            df.loc[task] = df.loc[task].apply(lambda x: x * 100 if x != -1 else -1)
    df['avg'] = df.mean(axis=1)
    df['avg.L'] = df.iloc[:,5:-1].mean(axis=1)
    combine_task_type(df)
    combine_task_type_2(df)
    df.to_csv(f'{OUTPUT_PATH}/{model_name}_{measurement}.csv')
def combine_task_type(df):
    classification = df.loc[classification_tasks+['BBH-geometric_shapes']].mean()
    translation = df.loc[translation_tasks].mean()
    summarization = df.loc[summarization_tasks].mean()
    math = df.loc[math_tasks].mean()
    science = df.loc[science_tasks].mean()
    symbolic = df.loc[[task for task in symbolic_tasks if task!="BBH-geometric_shapes"]].mean()
    df.loc['Classification'] = classification
    df.loc['Translation'] = translation
    df.loc['Summarization'] = summarization
    df.loc['Math'] = math
    df.loc['Science'] = science
    df.loc['Symbolic'] = symbolic
    return df
def combine_task_type_2(df):
    ASL = df.loc[ASL_tasks].replace(-1, np.nan).mean()
    SSL = df.loc[SSL_tasks].replace(-1, np.nan).mean()
    df.loc['ASL'] = ASL
    df.loc['SSL'] = SSL
    return df
def combine_task_type_results(models, task_types):
    for task_type in task_types:
        combined_df = pd.DataFrame()
        for model_name in models:
            df = pd.read_csv(f'{OUTPUT_PATH}/{model_name}_full_200.csv')
            df = df.set_index("Task")
            combined_df[model_name] = df.loc[task_type]
        combined_df = combined_df.T
        combined_df.index.name = 'Task'
        combined_df.to_csv(f'{OUTPUT_PATH}/{task_type}_full.csv')

if __name__ == '__main__':
    context_lengths = ['1k', '2k', '4k', '8k', '16k', '32k', '64k', '128k']
    create_csv_file(tasks, "Llama-3.1-8B-Instruct", "full", context_lengths)