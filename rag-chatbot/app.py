import streamlit as st

st.set_page_config(page_title="RAG Assistant", layout="centered")

# Initialize session state variables
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.title("🧠💬 RAG Assistant")

# Chat rendering
for chat in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(chat["question"])
    with st.chat_message("assistant"):
        st.markdown(chat["answer"])

# Input field
user_input = st.chat_input("Type your question here...", key="chat_input")

if user_input:
    # Append chat to history
    st.session_state.chat_history.append({
        "question": user_input,
        "answer": "🤖 (Mock Answer) This is a response from your RAG system."
    })
    
    # Clear the input field before rerun
    st.session_state.chat_input = ""

    # Optional rerun to refresh chat
    st.rerun()
