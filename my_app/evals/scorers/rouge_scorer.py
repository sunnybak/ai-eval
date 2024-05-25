from ai_eval import scorer
from rouge_score import rouge_scorer

def token_rouge(string_a, string_b):
    # str_a_list = get_str_token_list(string_a)
    # str_b_list = get_str_token_list(string_b)
    
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = scorer.score(string_a, string_b)
    return scores['rougeL'].precision



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
