from typing import Any, List
import pickle
import hashlib

class Score(object):
    def __init__(self, score, scorer_args=None, scorer_kwargs=None):
        self.score = score
        self.scorer_args = scorer_args
        self.scorer_kwargs = scorer_kwargs
        self.duration = None

    def __repr__(self):
        return f'{self.score}'

# this contains a list of scores

# eval = Evaluator([0.7, 1], lambda pass_range, score: pass_range[0] <= score <= pass_range[1])    
# eval(0.8) -> True
# eval(0.6, [0.3, 0.7]) -> True
# eval.in_range(0.6, [0.3, 0.7]) -> True
# 0.6 in eval.pass_range -> True

# if in_range is not provided, it defaults to lambda score: score == pass_range
# eval = Evaluator(pass_range=0.5)
# eval(0.5) -> True

# if pass_range is not provided, it defaults to True
# eval = Evaluator(in_range=lambda score: score == 1)
# eval(1) -> True
class Evaluator(object):
    def __init__(self, pass_range=None, in_range=None):
        self._pass_range = pass_range or True
        self._in_range = in_range or self._default_in_range
        self._result = None
        self._hash = None
    
    # default in case in_range is not provided
    def _default_in_range(self, score):
        return score == self._pass_range

    def __call__(self, *args, **kwds):
        result =  self._in_range(self._pass_range, *args, **kwds)
        self._result = (args, result)
        return result
    
    def __repr__(self) -> str:
        return f'{self._pass_range}'
