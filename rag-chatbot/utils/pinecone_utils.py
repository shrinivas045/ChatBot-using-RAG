import streamlit as st
from pinecone import Pinecone, ServerlessSpec

class PineconeClient:
    def __init__(self, index_name="rag-index", dimension=384):
        self.pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
        self.index_name = index_name

        # Create index if it doesn't exist
        if self.index_name not in self.pc.list_indexes().names():
            index_config = self.pc.create_index(
                name=index_name,
                dimension=dimension,
                metric='cosine',
                spec=ServerlessSpec(cloud='aws', region='us-east-1')
            )
            self.index = self.pc.Index(host=index_config.host)
        else:
            # Get the host for the existing index
            index_description = self.pc.describe_index(index_name)
            self.index = self.pc.Index(host=index_description.host)

    def upsert(self, vectors):
        # Convert dicts to tuples as required by Pinecone SDK 7.x
        pinecone_vectors = [
            (v['id'], v['values'], v.get('metadata', {})) for v in vectors
        ]
        self.index.upsert(vectors=pinecone_vectors)

    def query(self, embedding, top_k=5):
        return self.index.query(vector=embedding, top_k=top_k, include_metadata=True)
