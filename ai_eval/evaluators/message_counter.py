from ai_eval import Evaluator, Target
from ai_eval.evaluators.targets import NumericRangeTarget
from typing import Any, Callable, Dict, Tuple, List

def message_counter(state: Dict[str, Any]):
    '''Returns the number of messages'''
    count = len(state.get('messages', []))
    if count < 5:
        return False
    return True
