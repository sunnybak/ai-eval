import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

import pytest
from my_app.tests.util import async_score, openai_call
from my_app.src.code_suggester import coder_backend as backend

from my_app.evals.scorers.code_scorers import score_code_assistant
from my_app.evals.evaluators.code_evals import code_eval

# make a test for testing the coder_agent


# should use the fixed code prompts to instruct an LLM to generate Python code snippets
# generates a new user message or None to terminate
def code_prompt_ai_generator(messages):
    SYNTH_USER_PROMPT = """
        You are a software engineering professor. 
        Write an instruction with a short coding problem in Python of varying difficulties and skills for your student.
        Ask the student to write a Python code snippet which should be executable.
        Make sure to include the problem statement and any constraints.
    """
    model = 'gpt-4o'
    
    # 3 back and forths
    if (len(messages) - 1) // 2 < 1:
        task_instr = openai_call(
                            SYNTH_USER_PROMPT, 
                            model)
        return task_instr
    return None

@async_score(consistency=2)
async def score_test_case(i, test_case, model) -> bool:

    # initialize messages
    messages = backend.get_init_messages()

    # chat simulation
    while True:
        # generate user message
        user_msg = code_prompt_ai_generator(messages)
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


class TestGroup:

    def test_code_executes(self):

        batch_scores = score_test_case(None)
        print("Batch scores:", batch_scores)

        # run eval
        eval_result = code_eval(batch_scores["gpt-3.5-turbo"])
        

        print("Scores:", batch_scores["gpt-3.5-turbo"][0].scorer_args)
        print("Result:", eval_result)
        assert eval_result == True
