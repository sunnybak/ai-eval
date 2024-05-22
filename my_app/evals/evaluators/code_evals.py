from my_app.tests.types import Evaluator
import numpy as np

code_eval = Evaluator(
    pass_range=(0.9, 1.0),
    in_range=lambda pass_range, scores: pass_range[0]
    <= np.mean(scores)
    <= pass_range[1],
)