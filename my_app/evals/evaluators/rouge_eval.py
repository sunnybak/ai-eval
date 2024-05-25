from ai_eval import Evaluator
import numpy as np

# eval = 0.7 <= avg(rouge) < 1.0
mean_rouge_eval = Evaluator(pass_range=0.7, 
                            in_range=lambda range, scores: 
                                range < np.mean(scores) <= 1.0)

def comp(pass_range, scores):
    print('inside comp')
    print(scores)
    print(scores['result_0'])
    print(scores['result_1'])
    return np.all(scores['result_0']) and np.all(scores['result_1'])

comp_eval = Evaluator(pass_range=True,
                      in_range=comp
                     )