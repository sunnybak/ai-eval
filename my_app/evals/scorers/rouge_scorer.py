from my_app.tests.util import token_rouge, scorer

# score = average rougeness of the generated tasks
@scorer
def scorer_add_gen(messages, tasks=None) -> float:
    if tasks is None: return True
    final_answer = messages[-1]['content'].lower()
    rouge_scores = []
    for task in tasks:
        score = token_rouge(final_answer, '[] ' + task.lower())
        rouge_scores.append(score)
    return sum(rouge_scores)/len(rouge_scores)

