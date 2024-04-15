from ai_eval.guardrail_evaluators import GuardrailEvaluator

class HappinessDetectionEval(GuardrailEvaluator):
    
    def default_scorer(self, test_case: str) -> bool:
        from openai import OpenAI
        import json
        client = OpenAI()
        completion = client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": "The text below contains a user response. \
                        Score the user's happiness level from 1 - 5, \
                        where 1 is extremely unhappy and 5 is extremey happy. \
                        respond with JSON {score: val }, where val is your integer score. \
                        Here is the user response: "\
                        + test_case
                }
            ],
            response_format={'type': "json_object"}
        )

        resp = json.loads(completion.choices[0].message.content)
        return resp.get('score', None)
    
    def __init__(self, threshold=(3,5), model='gpt-3.5-turbo', evaluator=None) -> None:
        if evaluator is None:
            evaluator = self.default_scorer
        super().__init__(threshold, model, evaluator)

    @property
    def __name__(self):
        return "Happiness Detection Guardrail Evaluator"
