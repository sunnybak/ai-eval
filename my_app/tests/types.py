from typing import Any, List
import pickle
import hashlib
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
    def __init__(self, score, scorer_args=None, scorer_kwargs=None):
        self.score = score
        self.scorer_args = scorer_args
        self.scorer_kwargs = scorer_kwargs
        
    # create function with returns class vars as a kv dict
    def to_dict(self):
        return {
            'score': self.score,
            'scorer_args': self.scorer_args,
            'scorer_kwargs': self.scorer_kwargs
        }
        
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
        self._result = None
        self._hash = None
    
    # default in case in_range is not provided
    def _default_in_range(self, score):
        return score == self._pass_range

    def __call__(self, score, *args, **kwds):
        result =  self._in_range(self._pass_range, score, *args, **kwds)
        self._result = (score, result)
        return result
    
    def pickle(self):
        path = '../evals'
        # get the abs path and then hash the given path
        # make sure to encode recursively all the files in the path
        hash = hashlib.sha256(path.encode()).hexdigest()
        self._hash = hash

        # pickle this object and store it in a dir
        with open('../evals/results/', 'wb') as f:
            pickle.dump(self, f)

    # eval = Evaluator.unpickle(file_path)
    @classmethod
    def unpickle(self, path):
        path = '../evals/results/rick.pickle'
        with open(path, 'rb') as f:
            return pickle.load(f)
    
    def __repr__(self) -> str:
        return f'{self._pass_range}'
