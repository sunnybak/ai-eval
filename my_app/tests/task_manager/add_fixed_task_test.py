import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from typing import List, Dict
import pytest
from my_app import task_manager_backend as backend
from my_app.tests.types import Dataset, Target
from my_app.tests.util import async_score, scorer
import numpy as np

# generates a new user message or None to terminate
def generator_add_fixed(messages):
    trajectory = [
        '[] task 1 [x] go to supermarket [] beep boop',
        'add task 4',
        'add task 5',
    ]
    # count the number of user messages in messages using lambda
    user_idx = len(list(filter(lambda x: x['role'] == 'user', messages)))
    if user_idx < len(trajectory):
        return trajectory[user_idx]
    return None

# score = 1 if all tasks are added correctly, 0 if not
@scorer
def scorer_add_fixed(messages) -> bool:
    final_answer = messages[-1]['content'].lower()
    return \
        '[] task 1' in final_answer and \
        '[x] go to supermarket' in final_answer and \
        '[] beep boop' in final_answer and \
        '[] task 4' in final_answer and \
        '[] task 5' in final_answer

# target = 0.7 <= avg(correctness) < 1.0
target = Target(target_range=0.7, 
                in_range=lambda range, scores: range < np.mean(scores) <= 1.0)


@pytest.mark.parametrize("test_case", Dataset(
    test_cases=[
        # (generator, scorer, target)
        (generator_add_fixed, scorer_add_fixed, target),
    ],
))
def test_add_fixed_tasks(test_case):

    @async_score(consistency=10, models=['gpt-3.5-turbo', 'gpt-4'])
    async def score_test_case(i, test_case, model) -> bool:
        # initialize messages
        messages = backend.get_init_messages()
        
        # chat simulation
        while True:
            # generate user message
            user_msg = test_case.generator(messages)       
            if user_msg is None: break # exit condition
            backend.add_message(messages, {"role": "user", "content": user_msg})
            
            # fetch app response
            llm_response = await backend.a_get_llm_message(messages, model=model, stream=False)
            backend.add_message(messages, {"role": "assistant", "content": llm_response})

        print(f'\tCompleted async consistency rep {i+1}.')

        # score the trajectory
        score = test_case.scorer(messages)
        # if score not in target:
        #     print(f'Failed for model {model}: {score} {messages}')
        return score

    scores_df: Dict[str, List[tuple]] = score_test_case(test_case)
    print('Scores:\n', scores_df)
    
    result_df = scores_df.apply(lambda score: score in test_case.target, axis=0)
    print('Results:\n', result_df)

