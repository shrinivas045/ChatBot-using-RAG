import streamlit as st
from utils.embedder import get_embedding
from utils.llm_utils import query_llm
import numpy as np

# App Title
st.markdown("<h1 style='text-align: center; color:#4F46E5;'>💬 RAG Assistant</h1>", unsafe_allow_html=True)

# Custom CSS for monocolor layout with spacing
st.markdown("""
    <style>
    .chat-container {
        display: flex;
        flex-direction: column;
        gap: 24px;
        margin-bottom: 30px;
    }
    .user-bubble {
        align-self: flex-end;
        background: #D0E8FF;
        color: #1E3A8A;
        padding: 12px 16px;
        border-radius: 12px;
        max-width: 75%;
        font-size: 16px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.08);
    }
    .bot-bubble {
        align-self: flex-start;
        background: #F1F1F1;
        color: #111827;
        padding: 12px 16px;
        border-radius: 12px;
        max-width: 75%;
        font-size: 16px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.08);
    }
    .input-container {
        margin-top: 20px;
        padding-top: 15px;
        border-top: 1px solid #e0e0e0;
    }
    </style>
""", unsafe_allow_html=True)

# Session state init
if "context_set" not in st.session_state:
    st.session_state.context_set = False
    st.session_state.context_text = ""
    st.session_state.context_chunks = []
    st.session_state.context_embeddings = []
    st.session_state.chat_history = []
    st.session_state.previous_questions = set()
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

# Display chat history
if st.session_state.chat_history:
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for entry in st.session_state.chat_history:
        st.markdown(f"<div class='user-bubble'>🧑‍💬 {entry['question']}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='bot-bubble'>🤖 {entry['answer']}</div>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Chat input
if st.session_state.context_set and st.session_state.show_input_box:
    with st.container():
        st.markdown('<div class="input-container">', unsafe_allow_html=True)
        user_input = st.text_input("Type your question here...", key="chat_input", value="")
        col1, col2 = st.columns([1, 1])
        with col1:
            send_button = st.button("Send", use_container_width=True)
        with col2:
            new_question_button = st.button("New Question", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        if send_button and user_input.strip():
            if user_input.strip() in st.session_state.previous_questions:
                st.warning("You've already asked this question.")
            else:
                try:
                    query_embedding = get_embedding(user_input)
                    relevant_context = ""
                    if st.session_state.context_embeddings:
                        def cosine_sim(a, b):
                            a = np.array(a)
                            b = np.array(b)
                            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

                        sims = [cosine_sim(query_embedding, emb) for emb in st.session_state.context_embeddings]
                        top_k = 3
                        top_indices = np.argsort(sims)[-top_k:][::-1]
                        relevant_chunks = [st.session_state.context_chunks[i] for i in top_indices if sims[i] > 0]
                        if relevant_chunks:
                            relevant_context = "\n".join(relevant_chunks)

                    prompt = f"Answer the question based on the context below:\n\nContext:\n{relevant_context}\n\nQuestion: {user_input}" if relevant_context else user_input
                    response = query_llm(prompt, provider="groq")

                    if 'error' in response:
                        st.error(f"Error: {response['error']}")
                    else:
                        answer = response['choices'][0]['message']['content'] if 'choices' in response else str(response)

                        st.session_state.chat_history.append({
                            "question": user_input,
                            "answer": answer
                        })
                        st.session_state.previous_questions.add(user_input.strip())
                        st.session_state.chat_input = ""  # Clear input after submission
                        st.rerun()

                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                    st.info("Please check your API keys and make sure they are valid.")

        # Reset logic for New Question
        if new_question_button:
            st.session_state.chat_input = ""  # Clear input field
            st.rerun()
