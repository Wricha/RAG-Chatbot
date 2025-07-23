import os
import requests
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_community.vectorstores import Pinecone as LangchainPinecone
from langchain_community.embeddings import HuggingFaceEmbeddings

# Create Pinecone client instanceload_dotenv()
load_dotenv()
api_key = os.getenv("PINECONE_API_KEY")

pc = Pinecone(api_key)

def init_pinecone():
    index_name = os.getenv("PINECONE_INDEX_NAME")
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=384,  # Replace with actual embedding dimension
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",  # or "gcp"
                region=os.getenv("PINECONE_ENVIRONMENT", "us-west-2")
            )
        )
    return pc.Index(index_name)  # return the index object

def upsert_chunks(index_name, chunks, embedder):
    index = pc.Index(index_name)
    for i, doc in enumerate(chunks):
        vector = embedder.embed(doc.page_content)
        index.upsert([
            (f"id-{i}", vector, {"text": doc.page_content})
        ])

def search_similar(index_name, query, embedder, top_k=5):
    index = pc.Index(index_name)
    query_vector = embedder.embed_query(query)
    results = index.query(vector=query_vector, top_k=top_k, include_metadata=True)
    return [match['metadata']['text'] for match in results['matches']]
