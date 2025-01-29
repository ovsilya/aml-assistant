# vector_store.py

from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
import io
import os
import tempfile
import time

# Pinecone setup
PINECONE_API_KEY = "pcsk_oH4Du_4NfvagbXjtFHAQUueaqWQvNi347chn8PVXBXcEyiNopECT6M6woxvBrVLVeVz2A"  # Replace with your Pinecone API key
PINECONE_ENVIRONMENT = "us-east-1"
INDEX_NAME = "aml-assistant"
SERVICE_ACCOUNT_FILE = 'aml-chat-56afaacfced8.json'
SCOPES = ['https://www.googleapis.com/auth/drive']
os.environ["OPENAI_API_KEY"] = "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"

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

def authenticate_drive_api():
    """Authenticate and create a Google Drive API client."""
    credentials = Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=SCOPES
    )
    return build('drive', 'v3', credentials=credentials)

def initialize_vector_store():
    """Initialize the Pinecone vector store."""
    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
    if INDEX_NAME not in existing_indexes:
        pc.create_index(
            name=INDEX_NAME,
            dimension=1536,  # Adjust based on your embedding model
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",  # Replace with your actual cloud provider
                region=PINECONE_ENVIRONMENT
            )
        )
        while not pc.describe_index(INDEX_NAME).status["ready"]:
            time.sleep(1)
    index = pc.Index(INDEX_NAME)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return PineconeVectorStore(index=index, embedding=embeddings)

def process_pdf(file_id, file_name, drive_service, vector_store):
    """Process a single PDF file, extract text, and add it to the vector store."""
    request = drive_service.files().get_media(fileId=file_id)
    file_stream = io.BytesIO()
    downloader = MediaIoBaseDownload(file_stream, request)

    done = False
    while not done:
        _, done = downloader.next_chunk()

    file_stream.seek(0)
    with tempfile.NamedTemporaryFile(delete=True, suffix=".pdf") as temp_pdf:
        temp_pdf.write(file_stream.read())
        temp_pdf.flush()
        loader = PyPDFLoader(temp_pdf.name)
        documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    split_docs = text_splitter.split_documents(
        [Document(page_content=doc.page_content, metadata={"file_id": file_id, "file_name": file_name}) for doc in documents]
    )
    vector_store.add_documents(split_docs)

def process_folder(folder_id):
    """Recursively process all PDFs in a folder."""
    drive_service = authenticate_drive_api()
    vector_store = initialize_vector_store()

    def fetch_and_process_files(folder_id):
        """Fetch files and traverse subfolders."""
        query = f"'{folder_id}' in parents and trashed=false"
        results = drive_service.files().list(q=query, fields="files(id, name, mimeType)").execute()
        files = results.get("files", [])

        for file in files:
            if file["mimeType"] == "application/pdf":
                print(f"Processing file: {file['name']} (ID: {file['id']})")
                process_pdf(file["id"], file["name"], drive_service, vector_store)
            elif file["mimeType"] == "application/vnd.google-apps.folder":
                fetch_and_process_files(file["id"])

    fetch_and_process_files(folder_id)
    print("Processing complete.")

if __name__ == "__main__":
    # Replace with your folder ID containing PDFs
    FOLDER_ID = "1q-FZyHbwl5wYCoZ_iwNgOuEssgwKY_f6"
    process_folder(FOLDER_ID)
