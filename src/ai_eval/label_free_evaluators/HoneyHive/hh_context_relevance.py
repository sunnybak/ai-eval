

class HoneyHiveContextRelevance(ContextRelevanceEvaluator):
    
    def __init__(self, scorer):
        self.scorer = scorer
    
    def __call__(self, user_input: str, retrieved_context: List[str]):
        eval_result = self.scorer(user_input, retrieved_context)
        super().sanitized_eval_result(eval_result)
    
    @property
    def __name__(self):
        return "HoneyHive Context Relevance Evaluator"