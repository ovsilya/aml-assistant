from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
import os

# Pinecone setup
PINECONE_API_KEY = "pcsk_oH4Du_4NfvagbXjtFHAQUueaqWQvNi347chn8PVXBXcEyiNopECT6M6woxvBrVLVeVz2A" 
PINECONE_ENVIRONMENT = "us-east-1"
INDEX_NAME = "aml-assistant"
os.environ["OPENAI_API_KEY"] = "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"
os.environ["PINECONE_API_KEY"] = "pcsk_oH4Du_4NfvagbXjtFHAQUueaqWQvNi347chn8PVXBXcEyiNopECT6M6woxvBrVLVeVz2A"

# Initialize Pinecone
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

# Initialize OpenAI model
chat = ChatOpenAI(
    openai_api_key=os.environ["OPENAI_API_KEY"],
    model='gpt-3.5-turbo'
)

messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="Hi AI, how are you today?"),
    AIMessage(content="I'm great thank you. How can I help you?")
]

def augment_prompt(vector_store, query: str):
    # get top 3 results from knowledge base
    results = vector_store.similarity_search(query)
    # get the text from the results
    source_knowledge = "\n".join([x.page_content for x in results])
    
    # feed into an augmented prompt
    augmented_prompt = f"""Using the contexts below, answer the query.

    Contexts:
    {source_knowledge}

    Query: {query}"""
    return augmented_prompt


if __name__ == "__main__":
    
    """Connect to the existing Pinecone vector store."""
    index = pc.Index(INDEX_NAME)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = PineconeVectorStore(index=index, embedding=embeddings)
    query = "What kind of fitness equipment is included into well-being coverage?"

    # create a new user prompt
    prompt = HumanMessage(content=augment_prompt(vector_store, query))
    # add to messages
    messages.append(prompt)
    res = chat.invoke(messages)
    
    print("################################################################################################")
    print(res.content)
    
