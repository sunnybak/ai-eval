from types import Evaluator
import numpy as np

code_eval = Evaluator(
    pass_range=(0.7, 1.0),
    in_range=lambda pass_range, score: pass_range[0]
    <= np.mean(score)
    <= pass_range[1],
)

