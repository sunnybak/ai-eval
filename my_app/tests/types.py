from typing import Any, List

class TestCase(object):
    def __init__(self, generator, scorer, evaluator, kwargs):
        self.generator = generator
        self.scorer = scorer
        self.evaluator = evaluator
        self.kwargs = kwargs

class Dataset(object):
    def __init__(self, test_cases):
        self.test_cases = [
            TestCase(test_case[0], test_case[1], test_case[2]) for test_case in test_cases
        ]

    def __iter__(self):
        for test_case in self.test_cases:
            yield test_case
            
class Score(object):
    def __init__(self, score, scorer_args, scorer_kwargs):
        self.score = score
        self.scorer_args = scorer_args
        self.scorer_kwargs = scorer_kwargs

    def __float__(self):
        return float(self.score)
    
    def __add__(self, other):
        if isinstance(other, Score) or isinstance(other, (int, float)):
            return float(self) + float(other)
        return NotImplemented
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def __repr__(self):
        return f'{self.score}'

# this contains a list of scores
class BatchScore(object):
    def __init__(self, scores: List[Score]):
        self.scores = scores

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
    
    # default in case in_range is not provided
    def _default_in_range(self, score):
        return score == self._pass_range

    def __call__(self, scores, *args: Any, **kwds: Any) -> Any:
        return self._in_range(self._pass_range, scores, *args, **kwds)
    
    def __repr__(self) -> str:
        return f'{self._pass_range}'
