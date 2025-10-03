import os
import logging
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from utils import CHUNK_SIZE, CHUNK_OVERLAP, DOCUMENTS_FOLDER, logger

def load_documents_from_folder(folder_path):
    """Load all documents from folder and subfolders"""
    all_documents = []

    if not os.path.exists(folder_path):
        logger.warning(f"Folder not found: {folder_path}")
        return all_documents

    supported_extensions = {'.pdf', '.txt', '.docx'}

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            file_ext = os.path.splitext(file_path)[1].lower()

            if file_ext in supported_extensions:
                try:
                    if file_ext == '.pdf':
                        loader = PyPDFLoader(file_path)
                    elif file_ext == '.txt':
                        loader = TextLoader(file_path)
                    elif file_ext == '.docx':
                        loader = Docx2txtLoader(file_path)

                    documents = loader.load()

                    # Add folder and file metadata
                    folder_name = os.path.basename(root)
                    for doc in documents:
                        doc.metadata['folder'] = folder_name
                        doc.metadata['file_name'] = file

                    all_documents.extend(documents)
                    logger.info(f"✅ Loaded {file} from {folder_name} ({len(documents)} docs)")

                except Exception as e:
                    logger.error(f"❌ Error loading {file_path}: {str(e)}")

    return all_documents

def process_documents():
    """Process all documents into chunks"""
    documents = load_documents_from_folder(DOCUMENTS_FOLDER)

    if not documents:
        raise ValueError("No documents found in the documents folder")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )

    chunks = text_splitter.split_documents(documents)

    # Folder statistics
    folder_stats = {}
    for doc in chunks:
        folder = doc.metadata.get('folder', 'Unknown')
        folder_stats[folder] = folder_stats.get(folder, 0) + 1

    logger.info(f"📑 Created {len(chunks)} chunks from {len(documents)} documents across {len(folder_stats)} folders")
    return chunks, folder_stats
