from ai_eval.scorers.callable_evaluator import CallableEvaluator

class LengthChecker(CallableEvaluator):
    
    def evaluate(self, test_case):
        return len(test_case)