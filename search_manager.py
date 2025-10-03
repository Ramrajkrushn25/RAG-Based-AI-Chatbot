import requests
from utils import SERPER_API_KEY, logger

def google_search(query, num_results=5):
    if not SERPER_API_KEY:
        raise ValueError("SERPER_API_KEY is not set")
    
    try:
        url = "https://google.serper.dev/search"
        payload = {"q": query, "num": num_results}
        headers = {'X-API-KEY': SERPER_API_KEY, 'Content-Type': 'application/json'}
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        
        results = []
        if 'organic' in data:
            for result in data['organic']:
                results.append({
                    'title': result.get('title', ''),
                    'link': result.get('link', ''),
                    'snippet': result.get('snippet', ''),
                    'position': result.get('position', 0)
                })
        if 'answerBox' in data:
            answer = data['answerBox']
            results.insert(0, {
                'title': 'Direct Answer',
                'link': answer.get('link', ''),
                'snippet': answer.get('answer', '') or answer.get('snippet', ''),
                'position': 0,
                'is_answer': True
            })
        logger.info(f"🌐 Found {len(results)} results for: {query}")
        return results
    except Exception as e:
        logger.error(f"❌ Google search error: {str(e)}")
        return []

def format_search_results(results):
    if not results:
        return "No relevant web results found."
    formatted = "🌐 **Web Search Results:**\n\n"
    for i, res in enumerate(results[:3], 1):
        if res.get('is_answer'):
            formatted += f"🎯 **Direct Answer:** {res['snippet']}\n\n"
        else:
            formatted += f"{i}. **{res['title']}**\n   🔗 {res['link']}\n   📝 {res['snippet']}\n\n"
    return formatted

def get_web_context(query):
    results = google_search(query)
    return format_search_results(results), results

def calculate_search_confidence(query, vector_results, web_results):
    q = query.lower()
    current_keywords = ['current', 'latest', 'recent', 'today', 'now', '2024', '2025', 'news', 'update', 'breaking']
    is_current = any(k in q for k in current_keywords)
    doc_conf = min(len(vector_results) * 25, 100) if vector_results else 0
    web_conf = 80 if web_results else 40
    if is_current:
        web_conf = min(web_conf + 20, 100)
        doc_conf = max(doc_conf - 30, 0)
    return {
        "vector_search": doc_conf,
        "web_search": web_conf,
        "hybrid": (doc_conf + web_conf) // 2
    }
