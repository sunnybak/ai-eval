# ai-eval

## what is ai-eval?
`ai-eval` is a pytest-wrapper that allows you to simulate & evaluate your AI applications.

## why build ai-eval?
there are many LLM eval frameworks that are already out there.
so why are we building another one?

1. **extensibility**

most eval frameworks don't allow much customization or composition on the threadpooling / building own metrics. They want you to use their own out of the box evals

2. **simplicity**

most eval SDKs are too strictly typed and require a lot of boilerplate to get started. We want to make it as easy as possible to get started with AI evals

3. **usefulness**

no one has simple utilities like concurrency, trajectory test cases, or rate limiting.
everyone assumes the evaluators & apps are deterministic, & single data points are good enough. 
in reality, you will need statistical analysis to make any claims

## how to use ai-eval?

to setup a test, we need to define 3 things:
- **generator**: a function that generates the input data
- **scorer**: a function that scores the output data
- **evaluator**: a function that checks if the scores are in your target range

## what's next? 
features
- [ ] optimal stopping policy
- [ ] versioning based on repo hash & git hash
- [ ] historical scores visualization
- [ ] mock tool calling servers 
- [ ] more templated scorers, evaluators, generators
- [ ] support for multiple models
- [ ] typescript version

docs
- [ ] examples, cookbooks, best practices, documentation
- [ ] metrics and reports

