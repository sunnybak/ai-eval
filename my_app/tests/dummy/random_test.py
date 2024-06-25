from ai_eval import Evaluator, Target, scorer, run_experiment
from ai_eval.targets import NumericRangeTarget
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


def app(func, *args, **kwargs):
    async def wrapper():
        # func(*args, **kwargs)
        # print the args
        print(args)
    return wrapper

# app(args, kwargs) -> [Score]
# app(args, kwargs)
async def run_app(_, model, len):
    res = generator()
    await asyncio.sleep(random.random())
    # res = await a_openai_call('write a 200 word essay on happiness')
    return [scorer_1(res), scorer_2(res)]


def test_dummy():
    
    scores_df = run_experiment(
        app=run_app,
        args=('arg1', 'arg2'),
        hyperparam_dict={
            'model': ['gpt-3.5-turbo', 'gpt-4'],
            'prompt_1_version': [0, 1, 2],
            'prompt_2_version': [0, 1],
            'chunk_size': [100, 200],
            'ocr_api_key': ['aws-textract', 'llama-index-pdf-parser'],
        },
        consistency=4
    )
    
    print('Scorer Args:\n',scores_df[['scorer_args_0', 'scorer_args_1']])
    
    print('Scores:\n',scores_df[['score_0', 'score_1']])
    
    def comp(pass_range, scores):
        all_result_0 = np.all(scores['result_0'])
        all_result_1 = np.all(scores['result_1'])
        return all_result_0 or all_result_1, all_result_0 and all_result_1

    
    rng_eval = NumericRangeTarget(pass_range='[0.5,)')
    scores_df['result_rng'] = scores_df['score_1'].apply(rng_eval)

    comp_target = Target(in_range=comp)
    gen_target = Target(pass_range=(0.5, 1, 'abc', []),
                        in_range=lambda range, scores: range[0] < np.mean(scores) <= range[1]
                        )
    
    gen_eval = Evaluator(target=gen_target, scorer=scorer_1)
    comp_eval = Evaluator(target=comp_target)
    
    scores_df['result_0'] = scores_df['score_1'].apply(gen_eval)
    
    print(0.5 in gen_eval)
    scores_df['result_0'] = scores_df['score_0'].apply(gen_target.with_custom_in_range(lambda _,score: score == 1))
    scores_df['result_1'] = scores_df['score_1'].apply(gen_target)
    scores_df['result_comp'] = scores_df[['result_0', 'result_1']].apply(comp_eval, axis=1)
    
    scores_df['result_0'] = scores_df['score_0'].apply(gen_target)
    
    print('Results:\n',scores_df[['result_0', 'result_1','result_rng','result_comp']])
    
    print(scores_df[['arg_0', 'model', 'len', 'duration']])
    # print cols
    print(scores_df.columns)
    print(scores_df.shape)

test_dummy()
    
    # scores_df['score'] = scores_df['data'].apply(score)
    # scores_df['result'] = scores_df['score'].apply(eval)
    
    # eval as a service
    # evalClient
    
    # RPC
    
    # class EvalPipeline:
    #     def __init__(self, scorer, evaluator):
    #         self.evaluators = evaluators
        
    #     def __call__(self, data):
    #         return self.eval(self.score(data))
    
    # Evaluator = Scorer + Target(target, in_range)?
    
    
    # Score the safety of this prompt / 10  

    # Is this prompt safe to execute?

    # test ! = guardrail ? = evaluation =
    
    # df = pd.DataFrame()
    # df['guardrail'] = df['data'].apply(score).apply(eval)
    
    # guardrails
    # label-free
    # with-label
    
