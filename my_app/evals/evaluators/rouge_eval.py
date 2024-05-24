from my_app.tests.types import Evaluator
import numpy as np

# eval = 0.7 <= avg(rouge) < 1.0
mean_rouge_eval = Evaluator(pass_range=0.7, 
                            in_range=lambda range, scores: range < np.mean(scores) <= 1.0)
