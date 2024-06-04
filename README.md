# ai-eval

## what is ai-eval?
`ai-eval` is a simulation-based Agent Development Framework which helps you iterate by
1. Generating insightful synthetic trajectories for testing
2. Evaluating the trajectories to select the one with most interesting results
3. Boostrapping your Agent architecture for rapid development

## why build ai-eval?
there are many LLM eval frameworks that are already out there.
so why are we building another one?

1. **Synthetic Dataset Generation Toolkit**

AI Applications are too open ended. Customers don't know what they want. So we need to simulate and understand where our application breaks.

No framework exists for generating synthetic trajectories. Since agents trajectories are variable, we help you increase your test surface area using synthetic trajectories that you can customize with your ideal user persona.

2. **Maximizing Developer Attention**

AI Applications can only be aligned by observing the most anomalous trajectories.

We use statistics to analyze multiple trajectories per run and show you the most interesting one first. Test cases that deviate most from the others or trigger/break evaluations are more interesting for development.

3. **Composable Abstractions**

Current frameworks lack the base abstractions to evaluate trajectories and also make it hard to customize the evaluators.

For the beginner, we offer several boostrapped stacks for chatbots and agents.
For the novice, we offer customization in Evaluators, App Settings, and Trajectory generators.
For the expert, we allow developers to build their own console and simulation logic.




## Quickstart

Input:

```
system_prompt = 'be a good legal assistant'
evaluator = TopicScoreEvaluator('tax credits')
user_profile = 'small business owner looking for tax advice'

run(system_prompt, evaluator, user_profile)
```

Output:

```
Total cost: 7.663e-05
Total tokens: 1175
Average TPS: 804.7948882714338
Elapsed time: 1.459999332902953

Scores:  [8, 8, 9, 6, 9]
Mean:  8.0
Most interesting Chat:  6

Chat:

 SYSTEM 

 be a good legal assistant 

 USER 

 Let's get straight to it. I'm the owner of a thriving artisanal bakery in the city, and I'm struggling to navigate the complexities of accounting and tax regulations as I look to expand my business. 

 ASSISTANT 

 I'm happy to help! As your legal assistant, I'll do my best to guide you through the process of balancing your books and staying on the right side of the tax authorities.
...

Let's get started and work together to get your bakery's finances in order! 

 USER 

 Thanks for getting straight to it! I appreciate your no-nonsense approach. Here's the rundown on my bakery's financial situation:

Revenues: I'm doing around $800,000 in annual sales, with a steady growth rate over the past few years.
...

 ASSISTANT 

 Let's dive into the details. 
 ...
Some general guidelines for grant reporting:

1. Reporting timelines: Ensure you're aware of the required reporting deadlines and submit your reports on time.
2. Required documents: Keep all necessary documents, such as receipt of grants, invoices, and expenses related to the grant.
3. Compliance: Ensure compliance with the terms and conditions of the grant, including any restrictions or restrictions on the use of the funds.
4. Tracking: Keep accurate records of all grant-related expenses, budget vs. actual, and budget variances to track grant performance.

Let me know if there's anything else you'd like to discuss or if there are any specific areas you'd like me to focus on. 
```

Now that you have the intersting trajectory, iterate on your system_prompt, user_profile, or evaluator!






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


Tests: Alignment ! 😡

App: Alignment ? 🧐

Evals: Alignment = 😇