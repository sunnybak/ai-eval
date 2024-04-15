from ai_eval.base_evaluator import BaseEvaluator
from ai_eval.eval_result import EvalResult
from ai_eval.types import ScoreType

class GuardrailEvaluator(BaseEvaluator):

    # GuardrailEvaluator must take at least one string param
    def __call__(self, test_case: str, **scorer_kwargs) -> EvalResult:
        # unevaluated result
        eval_result = self.scorer(test_case, **scorer_kwargs)
        
        # evaluated result
        checked_threshold = self.check_threshold(eval_result)

        return checked_threshold
    
    def default_scorer(self, test_case: str) -> ScoreType:
        return test_case == True
    
    def __init__(self, threshold=0.5, model='gpt-3.5-turbo', scorer=None) -> None:
        if scorer is None:
            scorer = self.default_scorer
        super().__init__(threshold, model, scorer)
    
    @property
    def __name__(self):
        return "Base Guardrail Evaluator"
