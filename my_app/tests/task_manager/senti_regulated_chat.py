import streamlit as st
from openai import OpenAI
import random
import time
# from ai_eval import AIEval
# from my_app.evals.scorers import *

# ai = AIEval()

client = OpenAI(api_key='sk-proj-Gxa6OeFmoePlz04nf33ST3BlbkFJ00amYaRJhwF7gE0pVLve')

st.title("Simple chat")

# SYSTEM_PROMPT = "Solve the user's problem."
CHEERFUL_SYSTEM_PROMPT = "Respond to the user's messages with a positive message about life."
SERIOUS_SYSTEM_PROMPT = "Respond to the user's messages with a negative message about life."

# Set a default model
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": CHEERFUL_SYSTEM_PROMPT}]

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    

# Stream
def response_generator(mock=False):
    if mock:
        response = random.choice(
            [
                "Hello there! How can I assist you today?",
                "Hi, human! Is there anything I can help you with?",
                "Do you need help?",
            ]
        )
        for word in response.split():
            yield word + " "
            time.sleep(0.05)
    else:
        stream = client.chat.completions.create(
            model=st.session_state["openai_model"],
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            stream=True,
        )
        yield stream

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if user_input := st.chat_input("What is up?"):

    # add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # push point to logger
    # ai.write_point(point='chat.user_response', data=user_input) # want this to stack
    # user_is_cheerful_result = ai.eval(scorer=SentiScorer(), 
    #                                      points=['chat.user_response'], 
    #                                      threshold=['cheerful', 'serious'],
    #                                      point='chat.user_senti.eval'
    #                                     )
        
    # if user_is_cheerful_result:
    #     if user_is_cheerful_result.score == 'cheerful':
    #         st.session_state.messages[0] = {'role': 'system', 'content': SERIOUS_SYSTEM_PROMPT}
    #     elif user_is_cheerful_result.score == 'serious':
    #         st.session_state.messages[0] = {'role': 'system', 'content': CHEERFUL_SYSTEM_PROMPT}
    # else:
    #     ai.write_point('chat.bad_senti_score', user_is_cheerful_result.score) # want this to group

    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model=st.session_state["openai_model"],
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            stream=True,
        )
        response = st.write_stream(stream)

    # ai.write_point(point='chat.ai_response', data=response)
    # assert ai.eval(scorer=SentiScorer(), 
    #         points=['chat.ai_response'], 
    #         threshold=['cheerful', 'serious'], 
    #         point='chat.ai_senti.eval')
    
    st.session_state.messages.append({"role": "assistant", "content": response})