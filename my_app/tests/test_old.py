import pytest
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pytest
import time
from openai import OpenAI

client = OpenAI(api_key='sk-proj-Gxa6OeFmoePlz04nf33ST3BlbkFJ00amYaRJhwF7gE0pVLve')

SYSTEM_PROMPT = """
    You are a standup assistant. 

    You will be given the user's list of tasks their current status.
    
    Ask the user which tasks they would like to update.
    
    Respond with the updated list of tasks. \
    Tasks which are completed are marked with a [x] and tasks which are incomplete are marked with a [].
    Do not change the formatting of the task list.
"""

def chat_no_synth(system_prompt, trajectory=[]):
    message_history = [{"role": "system", "content": system_prompt}]
    counter = 0
    
    for user_message in trajectory:
        message_history.append({"role": "user", "content": user_message})
        ai_response = client.chat.completions.create(
            model='gpt-4o',
            messages=message_history,
        )
        message_history.append(ai_response.choices[0].message)
        counter += 1
    
    return message_history

# def test_updating():
#     for i in range(100):
#         trajectory = [
#             '[] task 1 [] task 2 [] task 3',
#             'mark task 3 as done',
#             'mark task 2 as done',
#             'mark task 3 as not done',
#             'mark task 1 as done',
#             'mark task 2 as not done',
#             'mark task 1 as not done',
#             'mark task 3 as done',
#         ]
        
#         conversation = chat_no_synth(SYSTEM_PROMPT, trajectory)
#         print('iteration ', i)
        
#         assert '[] task 1' in conversation[-1].content
#         assert '[] task 2' in conversation[-1].content
#         assert '[x] task 3' in conversation[-1].content

# def test_adding():
#     for i in range(100):
#         trajectory = [
#             '[] task 1 [x] task 2 [] task 3',
#             'add task 4',
#             'add task 5',
#         ]
        
#         conversation = chat_no_synth(SYSTEM_PROMPT, trajectory)
#         print('iteration ', i)
        
#         assert '[] task 1' in conversation[-1].content
#         assert '[x] task 2' in conversation[-1].content
#         assert '[] task 3' in conversation[-1].content
#         assert '[] task 4' in conversation[-1].content
#         assert '[] task 5' in conversation[-1].content


def test_adding_updating():
    for i in range(100):
        trajectory = [
            '[] task 1 [x] go to supermarket [] beep boop',
            'add task 4',
            'add task 5',
            'mark task 1 as done',
            'I have beep booped today',
            'mark task 4 as done',
            'remove task 1',
            'I did not go to the supermarket',
            'mark task 5 as done',
            'remove task 4',
        ]
        
        conversation = chat_no_synth(SYSTEM_PROMPT, trajectory)
        print('iteration ', i)
        
        assert '[] task 1' not in conversation[-1].content
        assert ('[] go to supermarket' in conversation[-1].content or '[] go to the supermarket' in conversation[-1].content)
        assert '[x] beep boop' in conversation[-1].content
        assert '[] task 4' not in conversation[-1].content
        assert '[x] task 5' in conversation[-1].content
