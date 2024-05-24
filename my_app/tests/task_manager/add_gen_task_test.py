import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from my_app.evals.evaluators.rouge_eval import mean_rouge_eval
from my_app.evals.scorers.rouge_scorer import scorer_add_gen
from my_app import task_manager_backend as backend
from my_app.tests.util import openai_call, token_rouge, batch_eval, scorer
import json


# generates a new user message or None to terminate
def generator_add_gen(messages, back_and_forth=3):
    SYNTH_USER_PROMPT = """
        Write a very short instruction (few words) for a task manager assistant to add a task to a list of tasks.
        For example, "add implement AI eval" or "I need to deploy fixes".
        Be creative with your language and task and try to come with a new instruction each time.
        From the instruction, extract the task verbatim and provide it as the task field in the JSON response.
        Respond strictly in the JSON format with the task and instruction containing the task, 
        like so: {"task": "optimize server", "instruction": "add a task called optimize server"}
    """
    model = 'gpt-4o'
    
    # back and forths
    if (len(messages) - 1) // 2 < back_and_forth:
        task_instr_json = openai_call(
                            SYNTH_USER_PROMPT, 
                            model,
                            response_format={'type': 'json_object'})
        return json.loads(task_instr_json)
    return None

async def run_score_test_case(_, model, back_and_forth=3, prompt_version=0):
    # initialize messages
    messages = backend.get_init_messages(prompt_version)
    
    # chat simulation 
    task_instr_json = generator_add_gen(messages, back_and_forth)
    user_msg = task_instr_json['instruction']
    tasks = [task_instr_json['task']]
    while user_msg is not None:
        backend.add_message(messages, {"role": "user", "content": user_msg})
        llm_response = await backend.a_get_llm_message(messages, model=model, stream=False)
        
        backend.add_message(messages, {"role": "assistant", "content": llm_response})
        task_instr_json = generator_add_gen(messages, back_and_forth)
        if task_instr_json is None: break
        user_msg = task_instr_json['instruction']
        tasks.append(task_instr_json['task'])

    # score the trajectory
    score = scorer_add_gen(messages, tasks=tasks)
    return score


def test_add_gen_tasks():
    
    hyperparam_dict = {
        'model': ['gpt-3.5-turbo', 'gpt-4'],
        'back_and_forth': [3, 10],
        'prompt_version': [0, 1]
    }

    scores_df = batch_eval(
        test_function=run_score_test_case,
        args=(None,),
        hyperparam_dict=hyperparam_dict,
        consistency=5
    )
    
    scores_df['result'] = scores_df['score'].apply(mean_rouge_eval)
    print(scores_df[['model','back_and_forth','prompt_version','score','result']])
    
    # save the scores to a file
    scores_df.to_csv('add_gen_tasks_scores.csv')


# guardrail = score(data) in target
# test = score(generator(data)) in target

# RAG
# llama index optimal chunking config for RAG
# naive knn pipeline
# data -> chunk and embed -> knn -> score -> target

# generator = query engine (chunk generator(chunk hyperparams))
# scorer = 

# data -> score -> target