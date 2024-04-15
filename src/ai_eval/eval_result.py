from typing import Dict, Optional

class EvalResult(object):
    score: Optional[float] = None
    success: Optional[bool] = None
    
    score_breakdown: Optional[Dict] = None
    reason: Optional[str] = None

    evaluation_model: Optional[str] = None
    
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
