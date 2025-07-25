import streamlit as st
from utils.embedder import get_embedding
from utils.llm_utils import query_llm
import numpy as np

# App title
st.markdown("<h1 style='text-align: center;'>🤖 RAG Assistant</h1>", unsafe_allow_html=True)

# Custom CSS for chat bubbles and layout
st.markdown("""
    <style>
    .user-bubble {
        background-color: #DCF8C6;
        padding: 10px 15px;
        border-radius: 15px;
        margin: 5px 0;
        width: fit-content;
        max-width: 80%;
        align-self: flex-end;
    }
    .bot-bubble {
        background-color: #F1F0F0;
        padding: 10px 15px;
        border-radius: 15px;
        margin: 5px 0;
        width: fit-content;
        max-width: 80%;
        align-self: flex-start;
    }
    .chat-container {
        display: flex;
        flex-direction: column;
    }
    .input-container {
        padding-top: 10px;
        margin-top: 10px;
        border-top: 1px solid #e6e6e6;
    }
    </style>
""", unsafe_allow_html=True)

# Session state initialization
if "context_set" not in st.session_state:
    st.session_state.context_set = False
    st.session_state.context_text = ""
    st.session_state.context_chunks = []
    st.session_state.context_embeddings = []
    st.session_state.chat_history = []

if "show_input_box" not in st.session_state:
    st.session_state.show_input_box = True

# Hardcoded context
if not st.session_state.context_set:
    hardcoded_context = """
    Streamlit is an open-source Python library that makes it easy to create and share custom web apps for machine learning and data science. 
    In just a few minutes you can build and deploy powerful data apps - so let’s get started!

    RAG stands for Retrieval-Augmented Generation. It combines document retrieval with a generative model like GPT to answer questions 
    based on specific context or documents. This approach improves accuracy and reduces hallucination.
    """
    context_chunks = [chunk.strip() for chunk in hardcoded_context.split('\n\n') if chunk.strip()]
    st.session_state.context_text = hardcoded_context
    st.session_state.context_chunks = context_chunks
    st.session_state.context_embeddings = [get_embedding(chunk) for chunk in context_chunks]
    st.session_state.context_set = True

# Display full chat history
if st.session_state.chat_history:
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for entry in st.session_state.chat_history:
        st.markdown(f"<div class='user-bubble'><strong>You:</strong> {entry['question']}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='bot-bubble'><strong>Bot:</strong> {entry['answer']}</div>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Chat input section
if st.session_state.context_set and st.session_state.show_input_box:
    with st.container
