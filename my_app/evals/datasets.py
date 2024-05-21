simple_dataset = [
    ("I am so happy", 'cheerful'),
    ("I am so stressed.", 'serious'),
]


def gen_chat():
    for message in simple_dataset:
        yield message

chat_dataset = [
    ("Hi, human! Is there anything I can help you with?", "exit"),
    ("Do you need help?", "exit"),
]


# Dataset Independent Vars
# gpt-4
# system prompts / user objectives
# few shot trajectory examples
# chat pipeline

# DatasetBuilder.with(user_objective=objective, style=style, examples=[], anomaly_detector=anomaly_detector, model=gpt-4)

# customer service app
# dataset = user objective = system prompt

# def synth_user_chat_pipeline(prompt):
#     message = chat_with_prompt(prompt)
#     yield message

__all__ = ['simple_dataset']