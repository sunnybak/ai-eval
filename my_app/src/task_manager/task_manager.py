import streamlit as st
from task_manager_backend import *

st.title("Simple chat")

# Set a default model
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = get_init_messages()
    
# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if user_input := st.chat_input("What is up?"):

    # add user message to chat history
    with st.chat_message("user"):
        st.markdown(user_input)
    add_message(st.session_state.messages, {"role": "user", "content": user_input})

    # get response from AI
    with st.chat_message("assistant"):
        stream = get_llm_message(st.session_state.messages, 
                                model=st.session_state["openai_model"],
                                stream=True)
        response = st.write_stream(stream)
    add_message(st.session_state.messages, {"role": "assistant", "content": response})