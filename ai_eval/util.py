from openai import OpenAI, AsyncOpenAI
import tiktoken

import asyncio
import itertools
import pandas as pd
import numpy as np
from typing import List, Dict
from ai_eval.eval_types import Score

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

async def a_openai_call_msg(messages=None, model='gpt-3.5-turbo'):
    if messages is None:
        raise ValueError("prompt must be provided")
    
    client = AsyncOpenAI(api_key="sk-proj-Gxa6OeFmoePlz04nf33ST3BlbkFJ00amYaRJhwF7gE0pVLve")
    completion = await client.chat.completions.create(
        model=model,
        messages=messages
    )
    return completion.choices[0].message.content

async def a_openai_call(prompt=None, model='gpt-3.5-turbo', response_format=None):
    if prompt is None:
        raise ValueError("prompt must be provided")
    
    client = AsyncOpenAI(api_key="sk-proj-Gxa6OeFmoePlz04nf33ST3BlbkFJ00amYaRJhwF7gE0pVLve")
    completion = await client.chat.completions.create(
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

import time

def batch_eval(test_function, args, hyperparam_dict: Dict[str, List], consistency=3):
    
    if not isinstance(args, tuple) or args is None or args == (None,):
        args = (None,)
    
    async def post_process(func, *args, **hyper_combo):
        start_time = time.time()
        
        # run the test function
        scores = await func(*args, **hyper_combo)

        # make sure the output is a list of Score objects
        score_list = scores if isinstance(scores, list) else [scores]
        score_obj_list = [score if isinstance(score, Score) else Score(score) for score in score_list]
        
        duration = round(time.time() - start_time, 3)
        return score_obj_list, duration

    async def run_tests():
        
        # assert that the hyperparam_dict keys are all strings and values are lists
        assert all([isinstance(k, str) for k in hyperparam_dict.keys()])
        assert all([isinstance(v, list) for v in hyperparam_dict.values()])
        
        # get the cartesian product of all hyperparam ranges
        hyperparam_kwargs = [dict(zip(hyperparam_dict, x)) for x in itertools.product(*hyperparam_dict.values())]

        score_tasks = []
        results_dict = []
        for hyper_combo in hyperparam_kwargs:
            result = {}
            
            # populate the args
            for i, arg in enumerate(args):
                result['arg_' + str(i)] = arg
            
            # populate the hyperparams
            for key, value in hyper_combo.items():
                result[key] = value

            results_dict.append(result)
            score_tasks.extend([asyncio.create_task(post_process(test_function, *args, **hyper_combo)) for _ in range(consistency)])

        print("\nTotal number of tests:", len(score_tasks))
        
        # add a callback function to show the progress
        for task in score_tasks:
            task.add_done_callback(lambda fut: print(f"\tTest {int(fut.get_name().split('-')[1])-1} completed"))

        # Wait for all tasks to complete and get the results
        task_results = await asyncio.gather(*score_tasks)
        score_lists, durations = zip(*task_results)
        
        # assert that all the elements of scores have the same length
        assert all([len(score_list) == len(score_lists[0]) for score_list in score_lists])
        
        # pack the score_obj_lists into batches
        batches = []
        batch = []
        for score_obj_list, duration in zip(score_lists, durations):
            batch.append((score_obj_list, duration))
            if len(batch) == consistency:
                batches.append(batch)
                batch = []
        
        # ensure that the batches are consistent
        assert len(results_dict) == len(batches)
        
        # unpack the batches
        # batches = [[(score1,duration1), (score2,duration2),...], [(score3,duration3), (score4,duration4)]
        # -> [[(score1, score2,...), (duration1, duration2,...)], [(score3, duration3), (score4, duration4)]]
        # -> ([score1, score2,...],[score3, score4,...]), ([duration1, duration2,...], [duration3, duration4,...])
        # = batched_score_lists, batched_durations
        batched_score_lists, batched_durations = zip(*[list(zip(*batch)) for batch in batches])
        
        # for each result dict, add the scorers and duration keys
        num_scores = len(batched_score_lists[0][0])
        for result_dict in results_dict:
            for i in range(num_scores):
                result_dict['score_' + str(i)] = []
                result_dict['scorer_args_' + str(i)] = []
                result_dict['scorer_kwargs_' + str(i)] = []

        ret_results_dict = []
        for result_dict, score_list_batch, duration_batch in zip(results_dict, 
                                                            batched_score_lists, 
                                                            batched_durations):
            result_dict['duration'] = list(duration_batch)
            
            # unpack each score list and group by batch index
            for score_list in score_list_batch:
                for i, score in enumerate(score_list):
                    result_dict['score_' + str(i)].append(score.score)
                    result_dict['scorer_args_' + str(i)].append(score.scorer_args)
                    result_dict['scorer_kwargs_' + str(i)].append(score.scorer_kwargs)

            ret_results_dict.append(result_dict)
        return pd.DataFrame(ret_results_dict)
    
    return asyncio.run(run_tests())


# scorer decorator to monitor the inputs and outputs of the scorer
def scorer(score_func):
    def wrapper(*args, **kwargs):
        score = score_func(*args, **kwargs)
        return Score(score, args, kwargs)
    return wrapper
