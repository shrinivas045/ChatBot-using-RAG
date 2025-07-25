import streamlit as st
from utils.embedder import get_embedding
from utils.llm_utils import query_llm
import numpy as np

st.title("RAG Assistant")

# Custom CSS
st.markdown("""
    <style>
    .link-button {
        background: none;
        border: none;
        color: #6c757d;
        text-decoration: underline;
        cursor: pointer;
        font-size: 1rem;
        padding: 0;
        margin: 0;
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
    for i, entry in enumerate(st.session_state.chat_history):
        st.markdown(f"**You:** {entry['question']}")
        st.markdown(f"🧠 **Bot:** {entry['answer']}")

# Chat input section
if st.session_state.context_set and st.session_state.show_input_box:
    user_input = st.text_input("Ask anything:", key="chat_input")
    if st.button("Send"):
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

            if relevant_context:
                prompt = f"Answer the question based on the context below:\n\nContext:\n{relevant_context}\n\nQuestion: {user_input}"
            else:
                prompt = user_input

            response = query_llm(prompt, provider="groq")

            if 'error' in response:
                st.error(f"Error: {response['error']}")
            else:
                if 'choices' in response:
                    answer = response['choices'][0]['message']['content']
                else:
                    answer = str(response)

                # Append to chat history
                st.session_state.chat_history.append({
                    "question": user_input,
                    "answer": answer
                })
                st.rerun()

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.info("Please check your API keys and make sure they are valid.")
