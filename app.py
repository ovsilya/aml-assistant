from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
import os

# Pinecone setup
PINECONE_API_KEY = "pcsk_oH4Du_4NfvagbXjtFHAQUueaqWQvNi347chn8PVXBXcEyiNopECT6M6woxvBrVLVeVz2A" 
PINECONE_ENVIRONMENT = "us-east-1"
INDEX_NAME = "aml-assistant"
os.environ["OPENAI_API_KEY"] = "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"

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

def load_vector_store():
    """Connect to the existing Pinecone vector store."""
    index = pc.Index(INDEX_NAME)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return PineconeVectorStore(index=index, embedding=embeddings)

def run_similarity_search(vector_store, query):
    """Perform a similarity search for the given query."""
    print(f"Running similarity search for query: {query}")
    similar_docs = vector_store.similarity_search(query)
    for i, doc in enumerate(similar_docs, 1):
        print(f"Result {i}: {doc.page_content}")

if __name__ == "__main__":
    vector_store = load_vector_store()

    # Replace with your test query
    test_query = "What is the XXX?"
    run_similarity_search(vector_store, test_query)
