from langchain.agents import Tool, AgentExecutor, create_tool_calling_agent
from langchain.prompts import PromptTemplate
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain.tools.retriever import create_retriever_tool
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
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

store = {}
system_prompt = read_file('system_prompt.txt')
# Initialize Pinecone
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

# Initialize OpenAI model
llm = ChatOpenAI(
    openai_api_key=os.environ["OPENAI_API_KEY"],
    model='gpt-3.5-turbo'
)

def read_file(file_name):
    """Helper function to read a file's content."""
    with open(file_name, 'r') as file:
        return file.read()

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

def log_chat_history(session_id: str):
    history = get_session_history(session_id)
    # logger.info(f"Chat history for session {session_id}: {history.messages}")
    # Alternatively, print the history:
    # print(f"Chat history for session {session_id}: {history.messages}")

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

"""Connect to the existing Pinecone vector store."""
index = pc.Index(INDEX_NAME)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = PineconeVectorStore(index=index, embedding=embeddings)

retriever = vector_store.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 5, "score_threshold": 0.3},
)

retriever_tool = create_retriever_tool(
    retriever,
    "Retriever",
    description="Retrieves information from the knowledge base."
)

tools = [retriever_tool]

prompt = PromptTemplate(
        template=system_prompt,
        input_variables=[   "input", 
                            # "context", ## no need, because we have retriever_tool_eng and retriever_tool_deu, which model uses.
                            "chat_history"]
    )

agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)

agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        return_intermediate_steps=True
    )

agent_with_chat_history = RunnableWithMessageHistory(
    agent_executor,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history"
)

def chat():
    user_message = "What kind of fitness equipment is included into well-being coverage?"
    user_id = 1

    result = agent_with_chat_history.invoke({"input": user_message}, config={"configurable": {"session_id": user_id}})
    response = result["output"]
    log_chat_history(user_id)
    return(response)

if __name__ == "__main__":
    chat()

    
    
