from ai_eval.base_evaluator import BaseEvaluator
from ai_eval.eval_result import EvalResult

class GuardrailEvaluator(BaseEvaluator):

    # GuardrailEvaluator calls only take the test_case param
    def __call__(self, test_case: str):
        eval_result = self.scorer(test_case)
        checked_threshold = super().check_threshold(eval_result)
        return checked_threshold
    
    def default_scorer(self, test_case: str) -> EvalResult:
        if self.scorer is None:
            raise NotImplementedError("default_scorer must be implemented if using a generic evaluator")
        return self.scorer(test_case)
    
    @property
    def __name__(self):
        return "Base Guardrail Evaluator"
