from dotenv import load_dotenv
import streamlit as st
from langchain.prompts import (
    ChatPromptTemplate,
)
import utils

st.set_page_config(page_title="Bible Talk", page_icon="🕊")
st.title("Bible Talk 🕊")
load_dotenv()

# Initialize chat history
if "chat_started" not in st.session_state:
    st.session_state.chat_started = True
    st.session_state.chat_history = []

# Load prompt and model to create chain
chain = utils.initialize_chain("gpt-4o-mini", prompt=ChatPromptTemplate.from_messages(utils.load_prompt()))

# Print the chat history (For every query input, streamlit refreshes the page)
if st.session_state.chat_started:
    if st.session_state.chat_history:
        for role, message in st.session_state.chat_history:
            with st.chat_message(role):
                st.markdown(message, unsafe_allow_html=True)

# Input the query
if query := st.chat_input():
    # Store the query
    st.session_state.chat_history.append(("user", query))
    # Print the query
    with st.chat_message("user"):
        st.markdown(query, unsafe_allow_html=True)

    # Invoke the chain
    result = chain.invoke(
        {
            "input": query,
            "history": []
        }
    )

    # Store the invoked answer
    st.session_state.chat_history.append(("assistant", result.content))
    # Print the answer
    with st.chat_message("assistant"):
        st.markdown(result.content, unsafe_allow_html=True)
