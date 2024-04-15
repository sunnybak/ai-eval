from typing import Dict, Optional
from ai_eval.types import ScoreType

class EvalResult(object):
    score: Optional[ScoreType] = None
    score_breakdown: Optional[Dict] = None 
    # [True, "happy", ..]
    # [True, False, ..]

    success: Optional[bool] = None
    reason: Optional[str] = None

    evaluation_model: Optional[str] = None
    
    def __init__(self, score: Optional[ScoreType] = None, success: Optional[bool] = None) -> None:
        if score is None: 
            return
        self.score = score
        self.success = success if success is not None else score == True
    
    def __bool__(self) -> bool:
        return self.success
    
    def __eq__(self, value: object) -> bool:
        return self.score == value
    
    def __lt__(self, value: object) -> bool:
        return self.score < value
    
    def __gt__(self, value: object) -> bool:
        return self.score > value
    
    def __le__(self, value: object) -> bool:
        return self.score <= value
    
    def __ge__(self, value: object) -> bool:
        return self.score >= value
    
    def __ne__(self, value: object) -> bool:
        return self.score != value
    
    def __str__(self) -> str:
        return f"Score: {self.score}, Success: {self.success}"
