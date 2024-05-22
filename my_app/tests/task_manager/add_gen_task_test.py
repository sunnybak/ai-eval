import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from typing import List, Dict
from my_app import task_manager_backend as backend
from my_app.tests.types import Dataset, Target
from my_app.tests.util import openai_call, token_rouge, async_score, scorer
import pytest
import json


# generates a new user message or None to terminate
def generator_add_gen(messages):
    SYNTH_USER_PROMPT = """
        Write a very short instruction (few words) for a task manager assistant to add a task to a list of tasks.
        For example, "add implement AI eval" or "I need to deploy fixes".
        Be creative with your language and task and try to come with a new instruction each time.
        From the instruction, extract the task verbatim and provide it as the task field in the JSON response.
        Respond strictly in the JSON format with the task and instruction containing the task, 
        like so: {"task": "optimize server", "instruction": "add a task called optimize server"}
    """
    model = 'gpt-4o'
    
    # 3 back and forths
    if (len(messages) - 1) // 2 < 3:
        task_instr_json = openai_call(
                            SYNTH_USER_PROMPT, 
                            model,
                            response_format={'type': 'json_object'})
        return json.loads(task_instr_json)
    return None

# score = average rougeness of the generated tasks
@scorer
def scorer_add_gen(messages, tasks=None) -> float:
    if tasks is None: return True
    final_answer = messages[-1]['content'].lower()
    rouge_scores = []
    for task in tasks:
        score = token_rouge(final_answer, '[] ' + task)
        rouge_scores.append(score)
    return sum(rouge_scores)/len(rouge_scores)

# target = 0.7 <= avg(rouge) < 1.0
def in_range(range, scores):
    return range < sum(scores)/len(scores) <= 1.0

target = Target(target_range=['happy', 'sad'], in_range=in_range)


@pytest.mark.parametrize("test_case", Dataset(
    test_cases=[
        # (generator, scorer, target)
        (generator_add_gen, scorer_add_gen, target),

    ],
))
def test_add_gen_tasks(test_case):

    @async_score(consistency=10, models=['gpt-3.5-turbo', 'gpt-4'])
    async def run_score_test_case(i, test_case, model):
        # initialize messages
        messages = backend.get_init_messages()
        
        # chat simulation 
        task_instr_json = test_case.generator(messages)
        user_msg = task_instr_json['instruction']
        tasks = [task_instr_json['task']]
        while user_msg is not None:
            backend.add_message(messages, {"role": "user", "content": user_msg})
            llm_response = await backend.a_get_llm_message(messages, model=model, stream=False)
            backend.add_message(messages, {"role": "assistant", "content": llm_response})
            task_instr_json = test_case.generator(messages)
            if task_instr_json is None: break
            user_msg = task_instr_json['instruction']
            tasks.append(task_instr_json['task'])
        print(f'\tCompleted async consistency run {i+1}.')
        # score the trajectory
        score = test_case.scorer(messages, tasks=tasks)
        # if [score.score] not in target:
        #     print(f'Failed for model {model}: {score}\n{messages}')
        return score

    agg_scores = run_score_test_case(test_case)
    print(agg_scores)
    # for model, scores in agg_scores.items():
    #     assert scores in test_case.target, f'Failed for model {model}: {scores}'



# guardrail = score(data) in target
# test = score(generator(data)) in target

# RAG
# llama index optimal chunking config for RAG
# naive knn pipeline
# data -> chunk and embed -> knn -> score -> target

# generator = query engine (chunk generator(chunk hyperparams))
# scorer = 

# data -> score -> target