from ai_eval.scorers.chat.chat_scorer import ChatScorer


def test_chat():
    messages = [
        { 'role': 'user', 'content': 'What is artificial intelligence?' },
        { 'role': 'assistant', 'content': 'Artificial intelligence is the simulation of human intelligence processes by machines, especially computer systems.' },
    ]
    
    chat_scorer = ChatScorer(messages)
    
    assert chat_scorer.total_chat_length() == 2
    assert chat_scorer.chat_role_counts() == {'user': 1, 'assistant': 1, 'system': 0}
    assert chat_scorer.chat_role_tokens() == {'user': 5, 'assistant': 17, 'system': 0}