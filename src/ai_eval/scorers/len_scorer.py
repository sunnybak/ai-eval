from ai_eval.scorers.callable_evaluator import CallableEvaluator

class StringLengthScore(CallableEvaluator):
    def evaluate(self, string) -> int:
        if not type(string) == str:
            raise TypeError("Arguments must be a string.")
        return len(string)