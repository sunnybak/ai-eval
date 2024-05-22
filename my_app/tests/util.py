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
                    tasks = [asyncio.create_task(scorer(i, test_case, model)) for i in range(consistency)]
                    model_scores = await asyncio.gather(*tasks)
                    model_score_dict[model] = model_scores
                return model_score_dict
            return asyncio.run(run_tests())
        return wrapper
    return decorator



# scorer decorator to monitor the inputs and outputs of the scorer
def scorer(score_func):
    def wrapper(*args, **kwargs):
        score = score_func(*args, **kwargs)
        return Score(score, args, kwargs)
    return wrapper
