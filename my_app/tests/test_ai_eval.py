from ai_eval import AIEval
import pytest


@pytest.fixture
def ai():
    return AIEval()

def test_push_point(ai):
    r = ai.write_point('chat.user_response', 'Hi')
    r = ai.read_point('chat.user_response')
    assert r == 'Hi'