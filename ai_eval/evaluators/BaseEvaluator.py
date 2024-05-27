from ai_eval.targets.BaseTarget import BaseTarget
from typing import Any, Callable, List
from ai_eval.scorers.score import Score

class BaseEvaluator(object):
    
    DEFAULT_TARGET = BaseTarget()
    DEFAULT_SCORER = lambda x: x
    
    def __init__(self, target: BaseTarget, scorer: Callable = None):
        self.target = target or BaseEvaluator.DEFAULT_TARGET
        self.scorer = scorer or BaseEvaluator.DEFAULT_SCORER
        
        self._scores: List[Score] = [] # cache of the scores for the target
        self._results: List[bool] = [] # cache of the results for the target
        
        
    # calling the evaluator runs the scorer and target.in_range(target, score)
    def __call__(self, *scorer_args: Any, **scorer_kwargs: Any) -> Any:
        print('Scorer Args:', scorer_args)
        # run the scorer
        score = self.scorer(*scorer_args, **scorer_kwargs)
        score_obj = score if isinstance(score, Score) else Score(score)
        self._scores.append(score_obj)

        # run the target
        result = self.target(score_obj.score)
        self._results.append(result)

        return result
    
    def __contains__(self, score: Any) -> bool:
        return self.target.__contains__(score)