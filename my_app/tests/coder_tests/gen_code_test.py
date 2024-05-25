from ai_eval.util import openai_call
from ai_eval import run_experiment

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


async def score_test_case(_, model) -> bool:

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

    # score the trajectory
    score = score_code_assistant(messages)
    return score


class TestGroup:

    def test_code_executes(self):

        scores_df = run_experiment(
            score_test_case,
            (None,),
            model='gpt-3.5-turbo',
            consistency=3,
        )
        print("Batch scores:", scores_df)

        scores_df['result'] = scores_df['score_0'].apply(code_eval)
        print(scores_df[['model', 'score_0', 'result']])