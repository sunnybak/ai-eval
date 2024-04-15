from ai_eval.guardrail_evaluators import GuardrailEvaluator
from ai_eval.guardrail_evaluators import *
from ai_eval.base_evaluator import CallableEvaluator

class ToneChecker(CallableEvaluator):
    def evaluate(self, test_case: str) -> bool:
        postive_ranges = ['happy', 'unhappy','dissatisfied', 'thrilled']
        for tone in postive_ranges:
            if tone in test_case:
                return tone
            
class EmotionalLevel(CallableEvaluator):
    def evaluate(self, test_case: str) -> bool:
        return test_case.count('so')

tone_eval = GuardrailEvaluator(threshold=['happy', 'excited', 'thrilled'], scorer=ToneChecker())
emo_eval = GuardrailEvaluator(threshold=[1,3], scorer=EmotionalLevel())

user_response = 'I am so so thrilled!'
happiness_level = int(tone_eval(user_response).success) * emo_eval(user_response).score

print('customer happiness level: ', happiness_level)

