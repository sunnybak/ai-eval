from ai_eval.guardrail_evaluators.guardrail import GuardrailEvaluator

class PromptInjectionDetectionEval(GuardrailEvaluator):
    
    def default_scorer(self, test_case: str) -> bool:
        raise NotImplementedError("Prompt Injection Guardrail Evaluator requires a scorer function to be passed in the constructor")
        
    def __init__(self, threshold=0.5, model='gpt-3.5-turbo', evaluator=None) -> None:
        if evaluator is None:
            evaluator = self.default_scorer
        super().__init__(threshold, model, evaluator)
    
    @property
    def __name__(self):
        return "Prompt Injection Detection Guardrail Evaluator"