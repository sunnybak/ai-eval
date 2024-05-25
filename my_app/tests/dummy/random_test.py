from ai_eval import Evaluator, scorer, run_experiment
from ai_eval.evaluators import NumericRangeEvaluator
import numpy as np
import random
import asyncio

def generator():
    return round(random.random(), 1)

@scorer
def scorer_1(x):
    return x

@scorer
def scorer_2(x):
    return x + 0.5


async def run_app(_, model, len):
    res = generator()
    await asyncio.sleep(random.random())
    # res = await a_openai_call('write a 200 word essay on happiness')
    return [scorer_1(res), scorer_2(res)]


def test_dummy():
    
    scores_df = run_experiment(
        app=run_app,
        args=(None,),
        hyperparam_dict={
            'model': ['gpt-3.5-turbo', 'gpt-4o', 'gpt-4o-turbo'],
            'len': [3, 10],
        },
        consistency=4
    )
    
    print('Scorer Args:\n',scores_df[['scorer_args_0', 'scorer_args_1']])
    
    print('Scores:\n',scores_df[['score_0', 'score_1']])
    
    def comp(pass_range, scores):
        all_result_0 = np.all(scores['result_0'])
        all_result_1 = np.all(scores['result_1'])
        return all_result_0 or all_result_1, all_result_0 and all_result_1


    
    rng_eval = NumericRangeEvaluator(pass_range='[0.5,)')
    scores_df['result_rng'] = scores_df['score_1'].apply(rng_eval)

    comp_eval = Evaluator(in_range=comp)
    gen_eval = Evaluator(pass_range=(0.5,1),
                        in_range=lambda range, scores: range[0] < np.mean(scores) <= range[1]
                        )
    scores_df['result_0'] = scores_df['score_0'].apply(gen_eval.with_custom_in_range(lambda _,score: score == 1))
    scores_df['result_1'] = scores_df['score_1'].apply(gen_eval)
    scores_df['result_comp'] = scores_df[['result_0', 'result_1']].apply(comp_eval, axis=1)
    
    print('Results:\n',scores_df[['result_0', 'result_1','result_rng','result_comp']])
    
    print(scores_df[['arg_0', 'model', 'len', 'duration']])
    # print cols
    print(scores_df.columns)
    print(scores_df.shape)
