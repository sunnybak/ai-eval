from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pytest
import time

# example - if the user is feeling sad, tell them life will be ok
# example - if the user is feeling happy, tell them to be more serious in life

# run the streamlit app with logging (user and AI messages are logged to redis)
# connect to streamlit app with selenium. Now you can read and write messages to the app

# the streamlit app will both log and use evals on those logs to adjust the system prompt
# we want to test how well this system works by measuring sentiment of the conversation
# so, we set up a pytest test
# the test will use the app as a simulated user who is feeling low
# it will use the eval log from the app and present it graphically

# app server
# logging server
# logging client
# pytest suite
# ai_eval
from ai_eval import AIEval
from my_app.evals.scorers import SentiScorer, ChihuahuaScorer
from my_app.evals.datasets import simple_dataset
from openai import OpenAI

ai = AIEval()

client = OpenAI(api_key='sk-proj-Gxa6OeFmoePlz04nf33ST3BlbkFJ00amYaRJhwF7gE0pVLve')

SYNTH_USER_PROMPT = "Act like a human talking to an AI chatbot about life. \
            Keep messages very short. Respond with a random statement about chihuahuas every time. \
            Your objective is to get the AI to respond with a statement about chihuahuas."

@pytest.fixture
def senti_scorer():
    return SentiScorer()

@pytest.mark.parametrize("message,expected_senti", simple_dataset)
def test_senti_eval(senti_scorer, message, expected_senti):
    assert senti_scorer([message]) == expected_senti

# @pytest.mark.skip
def test_chat_app():

    driver = webdriver.Chrome()  
    driver.get("http://localhost:8501")
    
    msgs = []
    textarea = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.TAG_NAME, 'textarea'))
    )

    # get all elements with the p tag
    p_tag = driver.find_elements(By.TAG_NAME, 'p')[0] # ignore this, it's the app's sys message
    msgs.append({"role": "system", "content": SYNTH_USER_PROMPT})

    while len(msgs) < 10:
        try:
            user_response = client.chat.completions.create(
                model='gpt-3.5-turbo',
                messages=[
                    {"role": m["role"], "content": m["content"]}
                    for m in msgs
                ],
            ).choices[0].message.content
            print('User response:', user_response)
            
            ai.write_point('chat.synth_user.eval', user_response)
            chi_eval = ai.eval(scorer=ChihuahuaScorer(), points=['chat.synth_user.eval'], threshold=['chihuahua'], point='chat.synth_user_chi.eval')
            assert chi_eval.success == True

            # submit user message 
            time.sleep(2)
            textarea.send_keys(user_response)
            driver.find_element(By.XPATH, "//button[@data-testid='stChatInputSubmitButton']").click()
            
            # wait until the length of p tags is greater than 1
            p_tags = driver.find_elements(By.TAG_NAME, 'p')
            WebDriverWait(driver, 10, 2).until(
                lambda driver: len(driver.find_elements(By.TAG_NAME, 'p')) > len(p_tags) + 1
            )
            
            # read the messages
            p_tags = driver.find_elements(By.TAG_NAME, 'p')[1:] # ignore app sys
            msgs = [{"role": "system", "content": SYNTH_USER_PROMPT}]
            for i, p_tag in enumerate(p_tags):
                if i % 2 == 0:
                    msgs.append({"role": "assistant", "content": p_tag.text})
                else:
                    msgs.append({"role": "user", "content": p_tag.text})

            print(msgs)
            
            assert ai.read_point('chat.senti_compare.eval') == "True"
        
            # assert ai.read_point('chat.ai_response') == 'Hi'
            # assert ai.read_point('chat.user_senti.eval') == True
            
            # response = WebDriverWait(driver, 10).until(
            #     EC.presence_of_element_located((By.TAG_NAME, 'p'))
            # )
            # assert "Hi" in response
            # ai.eval(scorer=CompareScorer(), points=['chat.user_senti.eval', 'chat.ai_senti.eval'], threshold=[True], point='chat.senti_compare.eval')
            
        finally:
            ai.write_point('chat.len', len(msgs))
    time.sleep(1000)

    # textarea = driver.find_element(By.XPATH, "//div[@data-testid='stMarkdownContainer']")
    # textarea = driver.find_element(By.TAG_NAME, 'p')
    # print(textarea)
    
    # time.sleep(2)

    # make a get request to http://127.0.0.1:5000/get-log
    # log_resp = requests.get('http://127.0.0.1:5000/getlog')
    
    # assert "Hi" in log_resp  # Assuming your app's title
    # driver.quit() 