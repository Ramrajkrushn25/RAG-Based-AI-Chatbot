import os
import logging
import shutil
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from utils import EMBEDDING_MODEL, VECTOR_STORE_PATH, TOP_K_RESULTS, logger

BATCH_SIZE = 500  # Create embeddings in batches to avoid memory issues

def initialize_embeddings():
    """Initialize CPU-efficient HuggingFace embeddings"""
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': False}
        )
        logger.info(f"✅ Initialized embeddings with model: {EMBEDDING_MODEL}")
        return embeddings
    except Exception as e:
        logger.error(f"❌ Error initializing embeddings: {str(e)}")
        raise

def create_vector_store(documents, embeddings):
    """Create vector store incrementally in batches"""
    try:
        # Clear existing store
        if os.path.exists(VECTOR_STORE_PATH):
            shutil.rmtree(VECTOR_STORE_PATH)

        vector_store = Chroma(
            persist_directory=VECTOR_STORE_PATH,
            embedding_function=embeddings
        )

        logger.info(f"Creating vector store in batches of {BATCH_SIZE}...")

        for i in range(0, len(documents), BATCH_SIZE):
            batch = documents[i:i + BATCH_SIZE]
            vector_store.add_documents(batch)
            logger.info(f"✅ Added batch {i // BATCH_SIZE + 1} ({len(batch)} docs)")

        vector_store.persist()
        logger.info(f"✅ Vector store created with {len(documents)} documents")
        return vector_store

    except Exception as e:
        logger.error(f"❌ Error creating vector store: {str(e)}")
        raise

def load_vector_store(embeddings):
    """Load existing vector store if available"""
    try:
        if not os.path.exists(VECTOR_STORE_PATH):
            return None

        vector_store = Chroma(
            persist_directory=VECTOR_STORE_PATH,
            embedding_function=embeddings
        )
        doc_count = vector_store._collection.count()
        if doc_count == 0:
            return None

        logger.info(f"✅ Loaded vector store with {doc_count} documents")
        return vector_store

    except Exception as e:
        logger.error(f"❌ Error loading vector store: {str(e)}")
        return None

def search_documents(vector_store, query, k=TOP_K_RESULTS):
    """Search for relevant documents"""
    try:
        if vector_store is None:
            return []

        results = vector_store.similarity_search_with_relevance_scores(query, k=k)
        filtered_results = [doc for doc, score in results if score > 0.6]
        return filtered_results

    except Exception as e:
        logger.error(f"❌ Error searching documents: {str(e)}")
        return []
