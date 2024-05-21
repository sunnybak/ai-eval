from ai_eval.evaluators.base_evaluator import BaseEvaluator
from ai_eval.eval_result import EvalResult
from typing import List

class AnswerCorrectnessEvaluator(BaseEvaluator):

    # AnswerCorrectnessEvaluator 
    def __call__(self, model_answer: str, golden_answer: str, **scorer_kwargs) -> EvalResult:
        
        # unevaluated result
        eval_result = self.scorer(model_answer, golden_answer, **scorer_kwargs)
        
        # evaluated result
        checked_threshold = self.check_threshold(self.threshold, eval_result)

        return checked_threshold
    
    def default_scorer(self, user_input: str, retrieved_context: List[str], **scorer_kwargs):
        return user_input == ''.join(retrieved_context)
    
    def __init__(self, threshold=0.5, model='gpt-3.5-turbo', scorer=None) -> None:
        if scorer is None:
            scorer = self.default_scorer
        super().__init__(threshold, model, scorer)
    
    @property
    def __name__(self):
        return "Base Answer Correctness Evaluator"
