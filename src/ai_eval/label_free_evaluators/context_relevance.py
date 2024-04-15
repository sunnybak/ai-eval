
from typing import List
from ai_eval.base_evaluator import BaseEvaluator

class ContextRelevanceEvaluator(BaseEvaluator):
        
    def __call__(self, user_input: str, retrieved_context: List[str]):
        eval_result = self.scorer(user_input, retrieved_context)
        super().sanitized_eval_result(eval_result)
    
    @property
    def __name__(self):
        return "Base Context Relevance Evaluator"
