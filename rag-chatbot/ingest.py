import streamlit as st
from utils.pinecone_utils import PineconeClient
from utils.embedder import get_embedding

if 'context_chunks' in st.session_state and st.session_state.context_chunks:
    if st.button("Ingest Context to Pinecone"):
        pc = PineconeClient()
        vectors = []
        for i, chunk in enumerate(st.session_state.context_chunks):
            embedding = get_embedding(chunk)
            vectors.append({
                'id': f'context_{i}',
                'values': embedding,
                'metadata': {'text': chunk}
            })
        pc.upsert(vectors)
        st.success(f"Successfully ingested {len(vectors)} context chunks into Pinecone.")
else:
    st.info("No context chunks found. Please enter context in the main app first.")
