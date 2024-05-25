from ai_eval.scorers import AbsBaseScorer

class RougeScorer(AbsBaseScorer):
    
    def rouge_score(
        self, target: str, prediction: str, score_type: str
    ) -> float:
        """Calculates the Rouge score for a given target and prediction.

        Rouge (Recall-Oriented Understudy for Gisting Evaluation) is a metric used for evaluating the quality of generated text,
        especially in tasks like text summarization.

        Args:
            target (str): The actual label or target text.
            prediction (str): The generated text from the model or LLM.
            score_type (str): The Rouge score type (Options: 'rouge1', 'rouge2', 'rougeL').

        Returns:
            float: The Rouge score for the given target and prediction, based on the specified score type.
        """
        try:
            from rouge_score import rouge_scorer
        except:
            pass

        assert score_type in [
            "rouge1",
            "rouge2",
            "rougeL",
        ], "score_type can be either rouge1, rouge2 or rougeL"
        scorer = rouge_scorer.RougeScorer([score_type], use_stemmer=True)
        scores = scorer.score(target, prediction)
        return scores[score_type].fmeasure
    

if __name__ == '__main__':
    rouge_scorer = RougeScorer()
    target = "programming"
    prediction = "program"
    score_type = "rougeL"
    rouge_score = rouge_scorer.rouge_score(target, prediction, score_type)
    print(f"Rouge score ({score_type}): {rouge_score}")