from abc import ABC, abstractmethod
from ai_eval.callable_evaluator import CallableEvaluator
from ai_eval.eval_result import EvalResult
from ai_eval.types import ThresholdType, ScoreType, CallableEvaluatorType
from typing import Optional, Union, Callable


class BaseEvaluator(ABC):
    
    model: Optional[str] = None
    threshold: ThresholdType
    scorer: Optional[CallableEvaluatorType]
    
    score: Optional[ScoreType] = None

    def __init__(self, 
                 threshold: ThresholdType = True, 
                 model: str = 'gpt-3.5-turbo', 
                 scorer: Optional[CallableEvaluatorType] = None
                ):
        if scorer is None or \
            not (callable(scorer) or issubclass(scorer.__class__, CallableEvaluator)):
            raise ValueError("scorer must be a defined function or a subclass of CallableEvaluator")
        
        self.threshold = threshold
        self.model = model
        self.scorer = scorer
    
    def default_scorer(self, *args):
        raise NotImplementedError("Subclasses must implement default_scorer method")
    
    def check_threshold(self, score: ScoreType) -> bool:
        eval_result = EvalResult()
        eval_result.evaluation_model = self.model
        eval_result.score = score
        
        # check equivalence if primitive types
        if type(self.threshold) == bool and type(score) == bool:
            eval_result.success = score == self.threshold
        elif type(self.threshold) == str and type(score) == str:
            eval_result.success = score == self.threshold
        elif (isinstance(self.threshold, (float, int)) and isinstance(score, (float, int))):
            eval_result.success = score == self.threshold
            
        # numeric range evaluation
        elif isinstance(self.threshold, (list, tuple)) and len(self.threshold) == 2 \
                and all(isinstance(i, (int, float)) for i in self.threshold):
            eval_result.success = self.threshold[0] <= score <= self.threshold[1]

        # check if string is in range of strings
        elif isinstance(self.threshold, (list, tuple)) and all(isinstance(i, str) for i in self.threshold):
            eval_result.success = score in self.threshold

        else:
            raise ValueError("Threshold and score outputs must be of same types and float/int, str, or bool")
        return eval_result
    
    @abstractmethod
    def __call__(self, *args, **kwargs) -> EvalResult:
        raise NotImplementedError("Subclasses must implement __call__ method")