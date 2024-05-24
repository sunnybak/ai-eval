from openai import OpenAI
import tiktoken
from rouge_score import rouge_scorer

import functools
import asyncio
import pandas as pd
import numpy as np
from typing import List, Dict
from my_app.tests.types import Score

def openai_call(prompt=None, model='gpt-3.5-turbo', response_format=None):
    if prompt is None:
        raise ValueError("prompt must be provided")
    
    client = OpenAI(api_key="sk-proj-Gxa6OeFmoePlz04nf33ST3BlbkFJ00amYaRJhwF7gE0pVLve")
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        response_format=response_format
    )
    return completion.choices[0].message.content

def cosine_similarity(string_a, string_b, model='text-embedding-3-small'):
    client = OpenAI(api_key="sk-proj-Gxa6OeFmoePlz04nf33ST3BlbkFJ00amYaRJhwF7gE0pVLve")
    vector_a = client.embeddings.create(input = [string_a], model=model).data[0].embedding
    vector_b = client.embeddings.create(input = [string_b], model=model).data[0].embedding
    dot_product = np.dot(vector_a, vector_b)
    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)
    similarity = dot_product / (norm_a * norm_b)
    return similarity

def get_str_token_list(text):
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens_integer = encoding.encode(text)
    tokens_bytes = [encoding.decode_single_token_bytes(token) for token in tokens_integer]
    str_list = [b.decode('utf-8').strip() for b in tokens_bytes]
    return str_list

def token_rouge(string_a, string_b):
    # str_a_list = get_str_token_list(string_a)
    # str_b_list = get_str_token_list(string_b)
    
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = scorer.score(string_a, string_b)
    return scores['rougeL'].precision

# consistency=20, # number of repetitions
# TODO: batch_size for number of tests to run in parallel
# models=['gpt-3.5-turbo', 'gpt-4', 'gpt-4o'], # models to sweep
# TODO: hyperparam sweep by iterating over cart prod of kwarg hyperparams
def async_score(consistency=3, models=['gpt-3.5-turbo']):
    def decorator(scorer):
        @functools.wraps(scorer)
        def wrapper(test_case):
            async def run_tests():
                model_score_dict: Dict[str, List[Score]] = {}
                for model in models:
                    print(f'\nRunning {consistency} tests using {model}.')
                    tasks = [asyncio.create_task(scorer(test_case, model)) for i in range(consistency)]
                    model_scores = await asyncio.gather(*tasks)
                    model_score_dict[model] = model_scores
                return model_score_dict
            return asyncio.run(run_tests())
        return wrapper
    return decorator

import asyncio
from typing import Dict, List
import itertools
import pandas as pd

def batch_eval(test_function, args, hyperparam_dict: Dict[str, List], consistency=3):
    async def run_tests():
        
        # assert that the hyperparam_dict keys are all strings and values are lists
        assert all([isinstance(k, str) for k in hyperparam_dict.keys()])
        assert all([isinstance(v, list) for v in hyperparam_dict.values()])
        
        # get the cartesian product of all hyperparam ranges
        hyperparam_kwargs = [dict(zip(hyperparam_dict, x)) for x in itertools.product(*hyperparam_dict.values())]
        print("Hyperparam kwargs:", hyperparam_kwargs)

        score_tasks = []
        results = []
        for hyper_combo in hyperparam_kwargs:
            result = {}
            
            # populate the args
            for i, arg in enumerate(args):
                result['arg_' + str(i)] = arg
            
            # populate the hyperparams
            for key, value in hyper_combo.items():
                result[key] = value

            results.append(result)
            score_tasks.extend([asyncio.create_task(test_function(*args, **hyper_combo)) for _ in range(consistency)])

        print("Total number of tests:", len(score_tasks))

        # Wait for all tasks to complete
        scores = await asyncio.gather(*score_tasks)
        
        batched_scores = []
        batch = []
        for i, score in enumerate(scores):
            score_obj = score if isinstance(score, Score) else Score(score)
            batch.append(score_obj)
            if len(batch) == consistency:
                batched_scores.append(batch)
                batch = []

        ret_results = []
        assert len(results) == len(batched_scores)
        for result, score_batch in zip(results, batched_scores):
            score_dict = {'score': [], 'scorer_args': [], 'scorer_kwargs': []}
            for score in score_batch:
                score_dict['score'].append(score.score)
                score_dict['scorer_args'].append(score.scorer_args)
                score_dict['scorer_kwargs'].append(score.scorer_kwargs)
            ret_results.append(dict(result, **score_dict))
        return pd.DataFrame(ret_results)
    
    return asyncio.run(run_tests())


# scorer decorator to monitor the inputs and outputs of the scorer
def scorer(score_func):
    def wrapper(*args, **kwargs):
        score = score_func(*args, **kwargs)
        return Score(score, args, kwargs)
    return wrapper
