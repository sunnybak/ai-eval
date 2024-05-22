import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

import pytest
from my_app.tests.util import async_score, openai_call
from my_app.src.code_suggester import coder_backend as backend

from my_app.evals.scorers.code_scorers import score_code_assistant
from my_app.evals.evaluators.code_evals import code_eval

# make a test for testing the coder_agent

SEED_TRAJECTORIES = [
    # ['Write a Python function that takes a list of integers as input and returns the sum of all the integers in the list.'],
    # ['Write a Python function that extracts all the occurances of the word AI in a given text.'],
    [
    """
        Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.
        You may assume that each input would have exactly one solution, and you may not use the same element twice.
        You can return the answer in any order.
    """
    ],
    [
    """
        Given two sorted arrays nums1 and nums2 of size m and n respectively, return the median of the two sorted arrays.
        The overall run time complexity should be O(log (m+n)).
    """
    ],
]

# should use the fixed code prompts to instruct an LLM to generate Python code snippets
def code_prompt_generator(messages, seed_trajectory):
    # count the number of user messages in messages using lambda
    user_idx = len(list(filter(lambda x: x["role"] == "user", messages)))
    if user_idx < len(seed_trajectory):
        return seed_trajectory[user_idx]
    else:
        return None

@async_score(consistency=2)
async def score_test_case(i, test_case, model) -> bool:
    seed_trajectory = test_case

    # initialize messages
    messages = backend.get_init_messages()

    # chat simulation
    while True:
        # generate user message
        user_msg = code_prompt_generator(messages, seed_trajectory)
        if user_msg is None:
            break  # exit condition
        backend.add_message(messages, {"role": "user", "content": user_msg})

        # fetch app response
        llm_response = await backend.a_get_llm_message(messages, model, stream=False)
        backend.add_message(messages, {"role": "assistant", "content": llm_response})

    print(f"\tCompleted async consistency rep {i+1}.")

    # score the trajectory
    score = score_code_assistant(messages)
    return score


@pytest.mark.parametrize("seed_trajectory", SEED_TRAJECTORIES)
class TestGroup:

    def test_code_executes(self, seed_trajectory):

        batch_scores = score_test_case(seed_trajectory)
        print("Batch scores:", batch_scores)

        # run eval
        eval_result = code_eval(batch_scores["gpt-3.5-turbo"])
        
        if eval_result == False:
            print("Failed test:", batch_scores["gpt-3.5-turbo"][0].scorer_args)
        print("Result:", eval_result)
        assert eval_result == True
    