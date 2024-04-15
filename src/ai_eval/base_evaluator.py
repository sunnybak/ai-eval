from abc import ABC, abstractmethod
from typing import Optional, Tuple
from ai_eval.scorers.callable_evaluator import CallableEvaluator
from ai_eval.eval_result import EvalResult
from typing import Optional, Tuple, Union, Callable, List

ThresholdType = Union[
    # primitive types
    bool, 
    int,
    float,
    str,
    
    # list of strings
    Tuple[str, ...],
    
    # numeric ranges
    Tuple[float, float], 
    Tuple[int, float],
    Tuple[float, int],
    Tuple[int, int]
]

class BaseEvaluator(ABC):
    
    model: Optional[str] = None
    threshold: ThresholdType
    scorer: Optional[Callable] | CallableEvaluator
    
    score: Optional[Union[float, int, str, bool]] = None

    def __init__(self, 
                 threshold: ThresholdType = (0.5, 1), 
                 model: str = 'gpt-3.5-turbo', 
                 scorer: Optional[Callable] = None
                ):
        if scorer is None or \
            not (callable(scorer) or issubclass(scorer.__class__, CallableEvaluator)):
            raise ValueError("scorer must be a defined function or a subclass of CallableEvaluator")
        
        self.threshold = threshold
        self.model = model
        self.scorer = scorer
    
    def check_threshold(self, score: Union[float, int, str, bool]) -> bool:
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
    def __call__(self, test_case: str):
        raise NotImplementedError("Subclasses must implement __call__ method")