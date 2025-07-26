import streamlit as st

st.set_page_config(page_title="RAG Assistant", layout="centered")

# Initialize session state variables
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.title("🧠💬 RAG Assistant")

# Display chat history
for chat in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(chat["question"])
    with st.chat_message("assistant"):
        st.markdown(chat["answer"])

# Input box
user_input = st.chat_input("Type your question here...")

if user_input:
    st.session_state.chat_history.append({
        "question": user_input,
        "answer": "🤖 (Mock Answer) This is a response from your RAG system."
    })

    # Just rerun without modifying chat_input manually
    st.experimental_rerun()
