from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.oauth2.service_account import Credentials
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders.pdf import PDFMinerLoader
import tempfile
from langchain_community.document_loaders import PyPDFLoader
import io
import os

# Define the scope for Google Drive API
SCOPES = ['https://www.googleapis.com/auth/drive']

# Path to your service account key
SERVICE_ACCOUNT_FILE = 'aml-chat-56afaacfced8.json'

def authenticate_drive_api():
    """Authenticate and create a Google Drive API client."""
    credentials = Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=SCOPES
    )
    return build('drive', 'v3', credentials=credentials)

def get_all_files_recursive(folder_id):
    """
    Recursively get all PDF files in a folder and its subfolders.

    Args:
        folder_id (str): The ID of the Google Drive folder.

    Returns:
        list: A list of file metadata dictionaries for PDF files.
    """
    drive_service = authenticate_drive_api()
    files = []
    query = f"'{folder_id}' in parents"
    
    # Recursive folder traversal
    def fetch_files(folder_id):
        nonlocal files
        query = f"'{folder_id}' in parents and trashed=false"
        results = drive_service.files().list(
            q=query, fields="files(id, name, mimeType)"
        ).execute()
        items = results.get('files', [])
        for item in items:
            if item['mimeType'] == 'application/pdf':
                files.append(item)  # Add PDF file to the list
            elif item['mimeType'] == 'application/vnd.google-apps.folder':
                fetch_files(item['id'])  # Recurse into subfolder

    fetch_files(folder_id)
    return files

def extract_text_from_drive_pdf(file_id):
    """
    Extract text from a PDF file directly from Google Drive.

    Args:
        file_id (str): The ID of the file on Google Drive.

    Returns:
        str: The extracted text.
    """
    drive_service = authenticate_drive_api()
    request = drive_service.files().get_media(fileId=file_id)
    file_stream = io.BytesIO()
    downloader = MediaIoBaseDownload(file_stream, request)
    done = False
    while not done:
        status, done = downloader.next_chunk()
        print(f"Download progress: {int(status.progress() * 100)}%")

    file_stream.seek(0)  # Reset the stream pointer for reading

    # Create a temporary file to store the PDF content
    with tempfile.NamedTemporaryFile(delete=True, suffix=".pdf") as temp_pdf:
        temp_pdf.write(file_stream.read())  # Write content to temp file
        temp_pdf.flush()  # Ensure all data is written to disk

        # Use PyPDFLoader to extract text
        loader = PyPDFLoader(temp_pdf.name)
        documents = loader.load()

    # Combine the text content of all pages
    return "\n".join(doc.page_content for doc in documents)

if __name__ == "__main__":
    # Replace with the ID of your Google Drive folder
    FOLDER_ID = "1q-FZyHbwl5wYCoZ_iwNgOuEssgwKY_f6"

    # Get all PDF files recursively in the folder and subfolders
    pdf_files = get_all_files_recursive(FOLDER_ID)

    if not pdf_files:
        print("No PDF files found in the specified folder and its subfolders.")
    else:
        print(f"Found {len(pdf_files)} PDF files in the folder and its subfolders.")

        for file in pdf_files:
            file_id = file['id']
            file_name = file['name']

            # Extract text from the PDF file
            print(f"Processing file: {file_name}")
            extracted_text = extract_text_from_drive_pdf(file_id)

            # Save the extracted text to a .txt file
            txt_file_name = file_name.replace('.pdf', '.txt')
            txt_file_path = os.path.join(os.getcwd(), txt_file_name)

            with open(txt_file_path, 'w', encoding='utf-8') as txt_file:
                txt_file.write(extracted_text)

            print(f"Extracted text saved to {txt_file_name}.")

        print("Processing completed.")
