import logging
from langchain_groq import ChatGroq
from utils import GROQ_API_KEY, logger, detect_query_type
from search_manager import get_web_context, calculate_search_confidence

def initialize_groq_llm():
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY environment variable is not set")
    try:
        llm = ChatGroq(
            groq_api_key=GROQ_API_KEY,
            model_name="llama-3.3-70b-versatile",
            temperature=0.1,
            max_tokens=2048
        )
        logger.info("✅ Initialized Groq LLM with model: llama-3.3-70b-versatile")
        return llm
    except Exception as e:
        logger.error(f"❌ Error initializing Groq LLM: {str(e)}")
        raise

def create_rag_prompt(query, context_documents, web_context, search_mode):
    doc_context = "\n\n".join([doc.page_content for doc in context_documents]) if context_documents else "No relevant documents found."
    
    mode_prompts = {
        "vector_search": f"""You are an AI research assistant. Answer based on the provided documents.

DOCUMENTS:
{doc_context}

QUESTION: {query}

INSTRUCTIONS:
- Answer strictly based on the documents
- If information isn't available, say so
- Be precise and factual""",

        "web_search": f"""You are an AI research assistant. Answer using web information.

WEB RESULTS:
{web_context}

QUESTION: {query}

INSTRUCTIONS:
- Use web results for current information
- Cite sources when possible
- Acknowledge limitations""",

        "hybrid": f"""You are an AI research assistant. Combine documents and web information.

DOCUMENTS:
{doc_context}

WEB RESULTS:
{web_context}

QUESTION: {query}

INSTRUCTIONS:
- First try documents, then supplement with web
- Prioritize recency for conflicting information
- Indicate sources clearly"""
    }
    return mode_prompts.get(search_mode, mode_prompts["hybrid"])

def generate_response(llm, query, context_documents, web_context, search_mode):
    try:
        prompt = create_rag_prompt(query, context_documents, web_context, search_mode)
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        logger.error(f"❌ Error generating response: {str(e)}")
        return f"Error: {str(e)}"

def automatic_search(llm, vector_store, query, chat_history=None):
    if chat_history is None:
        chat_history = []
    detected_mode = detect_query_type(query)
    vector_results, web_context, web_results = [], "", []
    
    if vector_store:
        from embedding_manager import search_documents
        vector_results = search_documents(vector_store, query)
    
    if detected_mode in ["web_search", "hybrid"]:
        web_context, web_results = get_web_context(query)
    
    confidence_scores = calculate_search_confidence(query, vector_results, web_results)
    final_mode = max(confidence_scores.items(), key=lambda x: x[1])[0]
    
    if len(vector_results) >= 2 and detected_mode != "web_search" and confidence_scores["vector_search"] > 60:
        final_mode = "vector_search"
    
    response = generate_response(llm, query, vector_results, web_context, final_mode)
    
    search_metadata = {
        "mode": final_mode,
        "confidence": confidence_scores[final_mode],
        "vector_results_count": len(vector_results),
        "web_results_count": len(web_results),
        "detected_mode": detected_mode,
        "confidence_scores": confidence_scores
    }
    
    chat_history.append({"question": query, "answer": response, "metadata": search_metadata})
    return response, chat_history, search_metadata
