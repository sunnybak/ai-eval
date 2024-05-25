from ai_eval.util import batch_eval, scorer
from ai_eval.eval_types import Evaluator
import numpy as np
import random
import time
import asyncio

def generator():
    return round(random.random(), 1)

@scorer
def scorer_1(x):
    return x

@scorer
def scorer_2(x):
    return x + 0.5


def comp(pass_range, scores):
    return np.all(scores['result_0']) or np.all(scores['result_1'])

comp_eval = Evaluator(in_range=comp)

gen_eval = Evaluator(pass_range=(0.5,1),
                        in_range=lambda range, scores: range[0] < np.mean(scores) <= range[1]
                    )

async def run_app(_, model, len):
    res = generator()
    await asyncio.sleep(random.random())
    # res = await a_openai_call('write a 200 word essay on happiness')
    return [scorer_1(res), scorer_2(res)]


def test_dummy():
    
    scores_df = batch_eval(
        test_function=run_app,
        args=(None,),
        hyperparam_dict={
            'model': ['gpt-3.5-turbo', 'gpt-4o', 'gpt-4o-turbo'],
            'len': [3, 10],
        },
        consistency=4
    )
    
    print('Scorer Args:\n',scores_df[['scorer_args_0', 'scorer_args_1']])
    
    print('Scores:\n',scores_df[['score_0', 'score_1']])

    scores_df['result_0'] = scores_df['score_0'].apply(gen_eval)
    scores_df['result_1'] = scores_df['score_1'].apply(gen_eval)
    scores_df['result_comp'] = scores_df[['result_0', 'result_1']].apply(comp_eval, axis=1)
    
    print('Results:\n',scores_df[['result_0', 'result_1','result_comp']])
    
    print(scores_df[['arg_0', 'model', 'len', 'duration']])
