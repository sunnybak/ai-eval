from abc import ABC, abstractmethod
from ai_eval.scorers.callable_scorer import CallableScorer
from ai_eval.eval_result import EvalResult
from ai_eval.types import ThresholdType, CallableScorerType
from typing import Optional, Union, Callable, Any

ScoreType = Union[float, int, str, bool]

class BaseEvaluator(ABC):
    
    model: Optional[str] = None
    threshold: ThresholdType
    scorer: Optional[CallableScorerType]
    normalizer: Optional[Any]
    
    score: Optional[ScoreType] = None

    def __init__(self, 
                 threshold: ThresholdType = True, 
                 model: str = 'gpt-3.5-turbo', 
                 scorer: Optional[CallableScorerType] = None
                ):
        if scorer is None or \
            not (callable(scorer) or issubclass(scorer.__class__, CallableScorer)):
            raise ValueError("scorer must be a defined function or a subclass of CallableScorer")
        
        self.threshold = threshold
        self.model = model
        self.scorer = scorer
    
    def default_scorer(self, *args):
        raise NotImplementedError("Subclasses must implement default_scorer method")
    
    def check_threshold(self, threshold, score: ScoreType) -> bool:
        eval_result = EvalResult()
        eval_result.evaluation_model = self.model
        eval_result.score = score
        
        # check equivalence if primitive types
        if type(threshold) == bool and type(score) == bool:
            eval_result.success = score == threshold
        elif type(threshold) == str and type(score) == str:
            eval_result.success = score == threshold
        elif (isinstance(threshold, (float, int)) and isinstance(score, (float, int))):
            eval_result.success = score == threshold
            
        # numeric range evaluation
        elif isinstance(threshold, (list, tuple)) and len(threshold) == 2 \
                and all(isinstance(i, (int, float)) for i in threshold):
            eval_result.success = threshold[0] <= score <= threshold[1]

        # check if string is in range of strings
        elif isinstance(threshold, (list, tuple)) and all(isinstance(i, str) for i in self.threshold):
            eval_result.success = score in self.threshold

        else:
            raise ValueError("Threshold and score outputs must be of same types and float/int, str, or bool. Received: ", type(self.threshold), type(score))
        return eval_result
    
    @abstractmethod
    def __call__(self, *args, **kwargs) -> EvalResult:
        raise NotImplementedError("Subclasses must implement __call__ method")
