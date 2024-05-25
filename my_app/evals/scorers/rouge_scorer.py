from ai_eval.util import scorer
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

@scorer
def scorer_human_or_ai(message, tasks=None):
    import random
    
    # 20% of the times ask for human input
    if random.random() < 0.2:
        # ask for human input
        # add it to queue
        return 0
    else:
        # ask for AI input
        
        return 1

# app
# evals
# tests