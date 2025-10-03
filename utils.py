import os
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
GROQ_MODEL = "llama-3.3-70b-versatile"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K_RESULTS = 4
VECTOR_STORE_PATH = "./vector_store"
DOCUMENTS_FOLDER = "documents"

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def validate_environment():
    """Validate required environment variables"""
    errors = []
    if not GROQ_API_KEY:
        errors.append("GROQ_API_KEY is not set")
    if not SERPER_API_KEY:
        errors.append("SERPER_API_KEY is not set")
    return errors

def detect_query_type(query):
    """Automatically detect the best search type for a query"""
    query_lower = query.lower()
    
    current_info_keywords = [
        'current', 'latest', 'recent', 'today', 'now', '2024', '2025',
        'news', 'update', 'breaking', 'trending', 'live', 'stock',
        'weather', 'crypto', 'bitcoin', 'price', 'rate'
    ]
    
    factual_keywords = [
        'what is', 'who is', 'when was', 'where is', 'how to',
        'definition', 'explain', 'meaning', 'history', 'facts'
    ]
    
    if any(keyword in query_lower for keyword in current_info_keywords):
        return "web_search"
    if any(keyword in query_lower for keyword in factual_keywords):
        return "hybrid"
    return "vector_search"

def get_mode_display_info(mode):
    """Get display information for each mode"""
    modes = {
        "vector_search": {
            "name": "Document Search",
            "description": "Searching through your knowledge base",
            "color": "#10B981",
            "icon": "📚"
        },
        "web_search": {
            "name": "Web Search", 
            "description": "Searching the web for current information",
            "color": "#3B82F6",
            "icon": "🌐"
        },
        "hybrid": {
            "name": "Hybrid Search",
            "description": "Combining knowledge base with web results",
            "color": "#8B5CF6",
            "icon": "🔗"
        }
    }
    return modes.get(mode, modes["vector_search"])
