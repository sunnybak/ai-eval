import pytest
from my_app import task_manager_backend as backend
from types import Evaluator
from ai_eval.util import batch_eval, scorer
import numpy as np


seed_trajectory = [
    [
        "[] task 1 [x] go to supermarket [] beep boop",
        "add task 4",
        "add task 5",
    ],
    # [
    #     "[] task 1 [x] go to supermarket [] beep boop",
    #     "addition task 4",
    #     "add task 5",
    # ],
    # [
    #     "[] task 1 [x] go to supermarket [] beep boop",
    #     "add task 4",
    #     "add task 5",
    # ],
]

scorer_checks = [
    [
        "[] task 1",
        "[x] go to supermarket",
        "[] beep boop",
        "[] task 4",
        "[] task 5",
    ],
    [
        "[] task 1",
        "[x] go to supermarket",
        "[] beep boop",
        "[] task 4",
        "[] task 5",
    ],
    [
        "[] task 1",
        "[x] go to supermarket",
        "[] beep boop",
        "[] task 4",
        "[] task 5",
    ],
]


def generator_add_fixed(messages, seed_trajectory):
    # count the number of user messages in messages using lambda
    user_idx = len(list(filter(lambda x: x["role"] == "user", messages)))
    if user_idx < len(seed_trajectory):
        return seed_trajectory[user_idx]
    return None


# score = 1 if all tasks are added correctly, 0 if not
@scorer
def scorer_add_fixed(messages, scorer_checks) -> bool:
    final_answer = messages[-1]["content"].lower()
    return all(check in final_answer for check in scorer_checks)

# target = 0.7 <= avg(correctness) < 1.0
eval = Evaluator(0.7, lambda pass_range, score: pass_range < np.mean(score) <= 1.0)


async def score_test_case(test_case, model) -> bool:
    seed_trajectory, scorer_checks = test_case
    
    # backend = Backend(configuration)
    
    # initialize messages
    messages = backend.get_init_messages()

    # chat simulation
    while True:
        # generate user message
        user_msg = generator_add_fixed(messages, seed_trajectory)
        if user_msg is None:
            break  # exit condition
        backend.add_message(messages, {"role": "user", "content": user_msg})

        # fetch app response
        llm_response = await backend.a_get_llm_message(
            messages, model=model, stream=False
        )
        backend.add_message(
            messages, {"role": "assistant", "content": llm_response}
        )

    # score the trajectory
    score = scorer_add_fixed(messages, scorer_checks)
    return score


@pytest.mark.parametrize(
    ("test_case"),
    [
        (seed_trajectory[i], scorer_checks[i]) for i in range(len(seed_trajectory))
    ],
)
class TestGroup:

    def test_add_fixed_tasks(self, test_case): 

        scores_df = batch_eval(test_function=score_test_case, 
                               args=(test_case,), 
                               hyperparam_dict={'model': ['gpt-3.5-turbo', 'gpt-4']}, 
                               consistency=2)

        scores_df['result'] = scores_df['score'].apply(eval)
        
        print(scores_df[['model', 'score', 'result']])

# test cases - regression, unittests
#   iterating: trajectories, generators
#   constant: app, scorer, target

# eval cases - guardrail, experiments
#   iterating: app, scorer, target
#   constant: trajectories, generators