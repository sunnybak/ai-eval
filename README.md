# ai-eval

Guardrail Evaluators
- Prompt Injection Detection
- Model Policy Violation Detection

Label Free Evaluators
- Context Relevance
- Answer Relevance
- Answer Faithfulness

With Label Evaluators
- Answer Correctness
- Context Correctness


ideas:
- decorator on evaluator method to aggregate results

The GenericScorer class has 2 key hyperparams
1. raw score (ScoreType)
2. norm score (ThresholdType)
When called with any arbitrary* input, they compute the
raw score and normalized score and return it

The input must be of the shape
- test_case (guardrail)

- user_input, model_answer
- user_input, retrieved_context
- user_input, retrieved_context, model_answer

- model_answer, golden_answer
- retrieved_context, golden_context

The actual parameters can be an object of any type.
The scorer returns a ScoreType object
The ScoreType is passed into the normalizer
The output of the normalizer is always of the type ThresholdType
This is so that the ThresholdTypes can be compared effectively
The default normalizer is simply the raw_score (assuming it is of ThresholdType)


The GenericEvaluator class has 2 key hyperparams
1. threshold (ThresholdType)
2. eval (func)
When called, they evaluate 2 ThresholdTypes
A custom evaluator can do advanced comparisons and checks between thresholds

scorer(EvalType) -> ScoreType 
normalizer(ScoreType) -> ThresholdType
evaluate(ThresholdType, ThresholdType) -> EvalResult


core framework
[] testing abstractions (concurrency, rate limits, consistency, dataset)
[] evaluators and scorers 
[] pipeline support (chat, agent, RAG)


[] examples, cookbooks, best practices, documentation
[] metrics and reports


[] model support

