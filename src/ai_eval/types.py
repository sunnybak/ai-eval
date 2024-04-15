from typing import Union, Tuple, Callable
from ai_eval.callable_evaluator import CallableEvaluator

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


ScoreType = Union[float, int, str, bool]

CallableEvaluatorType = Union[Callable, CallableEvaluator]