# AML Assistant - RAG-Based Compliance Chatbot

## Overview
The **AML Assistant** is a Retrieval-Augmented Generation (RAG) system designed for the **Swiss Banking industry** to help compliance officers and professionals answer queries related to **Anti-Money Laundering (AML)** and **Sanctions regulations**. The system leverages **LLMs** and **vector search** to retrieve relevant information from a knowledge base and provide intelligent responses.

## Features
- **RAG (Retrieval-Augmented Generation) architecture** for improved contextual answers.
- **Vector store similarity search** using **Pinecone** for fast and efficient information retrieval.
- **Integration with OpenAI's GPT-4o** for natural language processing.
- **Chat history management** for maintaining user session context.
- **PDF document ingestion and processing** from Google Drive.
- **Custom prompt engineering** for compliance-related queries.
- **Tool-based agent interaction** for dynamic and intelligent responses.

## Technology Stack
- **Programming Language:** Python
- **Frameworks/Libraries:**
  - LangChain (Agents, Tools, Runnables)
  - OpenAI (LLM & Embeddings)
  - Pinecone (Vector Database)
  - Flask (for API Deployment)
- **Infrastructure:**
  - Google Drive (for document storage)
  - Google Cloud Platform (GCP) (Deployment & Storage)

## Installation
### Prerequisites
Ensure you have Python 3.8+ installed and set up a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate   # On MacOS/Linux
venv\Scripts\activate     # On Windows
```

### Clone the Repository
```bash
git clone <repository_url>
cd <repository_directory>
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

## Environment Variables
Create a `.env` file or set environment variables manually:
```bash
export OPENAI_API_KEY='your-openai-api-key'
export PINECONE_API_KEY='your-pinecone-api-key'
export PINECONE_ENVIRONMENT='us-east-1'
```

## Pinecone Setup
The system uses Pinecone for vector storage. Ensure you have a Pinecone index created and update the environment variables accordingly.
```python
from pinecone import Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("aml-assistant")
```

## Running the Application
To start the chatbot, run:
```bash
python main.py
```

## Project Structure
```
├── data/                     # Folder containing processed knowledge base files
├── src/
│   ├── main.py               # Main chatbot script
│   ├── utils.py              # Utility functions
│   ├── retriever.py          # Vector store and retriever logic
│   ├── agent.py              # LangChain agent setup
│   ├── chat_history.py       # Chat session management
│   ├── system_prompt_main.txt # Custom system prompt
├── requirements.txt          # List of dependencies
├── README.md                 # Project documentation
```

## How It Works
1. **Data Processing:** PDFs are stored in Google Drive, converted to text, and stored in a vector database (Pinecone).
2. **Query Handling:** The chatbot processes user queries using LangChain’s retriever tool to fetch relevant knowledge.
3. **Response Generation:** The LLM (GPT-4o) generates responses based on retrieved knowledge and chat history.

## Example Query
```python
user_message = "Is caviar mentioned in the SECO regulation?"
result = agent_with_chat_history.invoke({"input": user_message}, config={"configurable": {"session_id": uuid.uuid4()}})
print(result["output"])
```

## Future Enhancements
- **Multi-language support (German, French, etc.)**
- **Improved document parsing techniques**
- **Real-time monitoring and analytics dashboard**

## Contributors
- **Ilya Ovsyannikov** (CTO at NavAI)
- NavAI Development Team

## License
This project is licensed under the MIT License. See `LICENSE` for more details.

