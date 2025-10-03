import streamlit as st
import time
from document_processor import process_documents
from embedding_manager import initialize_embeddings, create_vector_store, load_vector_store
from chat_manager import initialize_groq_llm, automatic_search
from utils import validate_environment, get_mode_display_info

# Page configuration (must be before other Streamlit UI calls)
st.set_page_config(
    page_title="NeuroSearch AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS (kept your theme) ---
st.markdown(
    """
<style>
/* ==========================
   Global body & fonts
=========================== */
body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    color: #f0f0f0;
    background: linear-gradient(135deg, #1e0033, #4b0066, #7d00a0); /* purple-magenta gradient */
}

/* ==========================
   Gradient heading
=========================== */
.gradient-text {
    background: linear-gradient(90deg, #f0c27b, #4b0082); /* gold to deep purple */
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 700;
}

/* ==========================
   Cards
=========================== */
.neuro-card {
    background: rgba(255, 255, 255, 0.05); /* semi-transparent glass */
    color: #f0f0f0;
    border-radius: 15px;
    padding: 1rem;
    margin-bottom: 1rem;
    box-shadow: 0 8px 20px rgba(0, 0, 0, 0.6);
    backdrop-filter: blur(10px);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.neuro-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 28px rgba(0,0,0,0.8);
}

/* ==========================
   Category headers for quick questions
=========================== */
.category-header {
    font-size: 1.2rem;
    font-weight: 600;
    color: #f0c27b; /* gold accent */
    margin-top: 1rem;
    margin-bottom: 0.5rem;
    border-left: 4px solid #f0c27b;
    padding-left: 0.5rem;
}

/* ==========================
   Chat messages
=========================== */
.user-message {
    background-color: #7d00a0; /* purple */
    color: #ffffff;
    padding: 0.7rem 1.2rem;
    border-radius: 15px;
    margin-bottom: 0.5rem;
    max-width: 85%;
    word-wrap: break-word;
}

.assistant-message {
    background-color: #f0c27b; /* gold */
    color: #1e0033; /* dark purple text */
    padding: 0.7rem 1.2rem;
    border-radius: 15px;
    margin-bottom: 0.5rem;
    max-width: 85%;
    word-wrap: break-word;
}

/* ==========================
   Mode indicator
=========================== */
.mode-indicator {
    border-left: 4px solid #f0c27b;
    padding-left: 0.6rem;
    margin-bottom: 0.5rem;
    background-color: rgba(255,255,255,0.05);
    border-radius: 8px;
    font-size: 0.9rem;
    color: #f0f0f0;
}

/* ==========================
   Stats numbers
=========================== */
.stat-number {
    font-size: 2rem;
    font-weight: 700;
    color: #f0c27b; /* gold accent */
    margin: 0.2rem 0;
}

/* ==========================
   Doc structure in welcome
=========================== */
.doc-structure h4 {
    margin-bottom: 0.3rem;
    color: #f0c27b;
}

.doc-structure p {
    margin: 0;
    padding-left: 0.5rem;
    color: #d1d5dc;
}

/* ==========================
   Progress bars inside analytics
=========================== */
.stProgress > div > div > div > div {
    background: linear-gradient(to right, #7d00a0, #f0c27b);
}

/* ==========================
   Links & hover effects
=========================== */
a {
    color: #f0c27b;
}

a:hover {
    text-decoration: underline;
}

/* ==========================
   Chat input box text color
=========================== */
input[type="text"], textarea {
    color: #f0f0f0 !important;
    background-color: rgba(255,255,255,0.05) !important;
    border-radius: 10px;
    border: 1px solid rgba(255,255,255,0.2);
}

/* ==========================
   Buttons text and background
=========================== */
.stButton>button {
    background: linear-gradient(90deg, #f0c27b, #7d00a0); /* gold → purple */
    color: #1e0033;
    font-weight: 600;
    border-radius: 12px;
    padding: 0.5rem 1.2rem;
    transition: background 0.3s ease, transform 0.2s ease;
}

.stButton>button:hover {
    background: linear-gradient(90deg, #7d00a0, #f0c27b); /* reverse gradient */
    color: #ffffff;
    transform: translateY(-2px);
}

</style>
""",
    unsafe_allow_html=True,
)

# ----------------------
# Session-state helpers
# ----------------------
def initialize_session_state():
    """Initialize session state variables"""
    defaults = {
        "vector_store": None,
        "llm": None,
        "chat_history": [],
        "documents_loaded": False,
        "embeddings": None,
        "current_search_mode": None,
        "system_initialized": False,
        "folder_stats": {},
        # helper for pending quick questions
        "pending_question": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

# ----------------------
# System initialization
# ----------------------
def initialize_system():
    """Initialize the entire system"""
    with st.spinner("🔮 Initializing NeuroSearch AI..."):
        try:
            # Initialize embeddings (only once)
            if st.session_state.embeddings is None:
                st.session_state.embeddings = initialize_embeddings()

            # Try to load existing vector store first (may return None)
            st.session_state.vector_store = load_vector_store(st.session_state.embeddings)

            if not st.session_state.vector_store:
                # Process documents from folder
                documents, folder_stats = process_documents()
                st.session_state.folder_stats = folder_stats or {}
                # Create vector store (documents may be list of dicts)
                st.session_state.vector_store = create_vector_store(documents, st.session_state.embeddings)
                st.session_state.documents_loaded = True
            else:
                st.session_state.documents_loaded = True
                st.session_state.folder_stats = {"Cached": "Previously loaded"}

            # Initialize LLM (Groq)
            if st.session_state.llm is None:
                st.session_state.llm = initialize_groq_llm()

            st.session_state.system_initialized = True
            return True, "System initialized successfully!"
        except Exception as e:
            # Return error message to UI
            return False, f"Initialization failed: {str(e)}"

# ----------------------
# UI helper functions
# ----------------------
def safe_get(dct, key, default=None):
    try:
        return dct.get(key, default)
    except Exception:
        return default

def display_search_analytics(metadata):
    """Display search analytics without plotly"""
    if not metadata:
        return

    # Safely extract values
    mode = safe_get(metadata, "mode", "unknown")
    confidence = safe_get(metadata, "confidence", 0)
    vector_results_count = safe_get(metadata, "vector_results_count", 0)
    web_results_count = safe_get(metadata, "web_results_count", 0)
    confidence_scores = safe_get(metadata, "confidence_scores", {})

    with st.expander("📊 Search Analytics", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Search Mode", str(mode).title())
        with c2:
            st.metric("Confidence", f"{confidence}%")
        with c3:
            st.metric("Documents", str(vector_results_count))
        with c4:
            st.metric("Web Sources", str(web_results_count))

        st.markdown("#### Confidence Scores")
        if isinstance(confidence_scores, dict) and confidence_scores:
            for m, score in confidence_scores.items():
                mode_display = get_mode_display_info(m)
                st.write(f"{mode_display.get('icon','')} **{mode_display.get('name',m)}**: {score}%")
                try:
                    st.progress(min(max(score / 100.0, 0.0), 1.0))
                except Exception:
                    pass
        else:
            st.write("Confidence breakdown not available.")

def display_quick_questions():
    """Display quick questions in an organized way"""
    st.markdown("### 💡 Quick Questions")

    categories = {
        "🤖 Artificial Intelligence": [
            "Explain the difference between AI and machine learning",
            "What are the latest trends in natural language processing?",
            "How do neural networks learn from data?",
            "What are the ethical considerations in AI development?"
        ],
        "💼 Business Strategy": [
            "What are the key elements of a successful business model?",
            "How to conduct competitive market analysis?",
            "Explain different digital transformation strategies",
            "What are current challenges in global business?"
        ],
        "🔬 Research Methodology": [
            "What are the best practices for academic research?",
            "How to design effective research experiments?",
            "Explain qualitative vs quantitative research methods",
            "What makes research findings statistically significant?"
        ],
        "🚀 Technology Trends": [
            "What are the emerging technologies in 2024?",
            "How is cloud computing evolving?",
            "Explain the impact of IoT on daily life",
            "What are the security challenges in modern tech?"
        ]
    }

    for category, questions in categories.items():
        st.markdown(f'<div class="category-header">{category}</div>', unsafe_allow_html=True)
        for question in questions:
            # Use stable key by using absolute hash
            key = f"q_{abs(hash(question))}"
            if st.button(question, key=key, use_container_width=True):
                st.session_state.pending_question = question
                # We rerun so the question flows into the chat_input processing
                st.rerun()

def display_welcome_screen():
    """Display welcome screen"""
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown('<div class="neuro-card">', unsafe_allow_html=True)
        st.markdown("### 🎯 Welcome to NeuroSearch AI")
        st.markdown("Your intelligent research companion that combines document analysis with real-time web intelligence.")
        # two-column features
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("""
            <div class="neuro-card" style="text-align: center;">
                <h3>📚 Smart Documents</h3>
                <p>AI-powered analysis of your knowledge base</p>
            </div>""", unsafe_allow_html=True)
            st.markdown("""
            <div class="neuro-card" style="text-align: center;">
                <h3>🔍 Adaptive Search</h3>
                <p>Automatically chooses the best search strategy</p>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown("""
            <div class="neuro-card" style="text-align: center;">
                <h3>🌐 Live Intelligence</h3>
                <p>Real-time web search integration</p>
            </div>""", unsafe_allow_html=True)
            st.markdown("""
            <div class="neuro-card" style="text-align: center;">
                <h3>⚡ Lightning Fast</h3>
                <p>Powered by Groq's ultra-fast inference</p>
            </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="neuro-card">', unsafe_allow_html=True)
        st.markdown("### 🚀 Getting Started")
        st.markdown("""
        1. Ensure your documents are in the `documents/` folder  
        2. Click 'Initialize System' in the sidebar  
        3. Start asking research questions!
        """)
        st.markdown("""
        <div class="doc-structure">
            <h4>📁 Your Document Structure</h4>
            <p>• Technology/</p>
            <p>• Business/</p>
            <p>• Research/</p>
            <p>• Company/</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ----------------------
# Main
# ----------------------
def main():
    # header
    h1_col_left, h1_col_mid, h1_col_right = st.columns([1, 2, 1])
    with h1_col_mid:
        st.markdown(
            '<h1 class="gradient-text" style="text-align: center; font-size: 3rem; margin-bottom: 0.5rem;">NeuroSearch AI</h1>',
            unsafe_allow_html=True
        )
        st.markdown('<p style="text-align: center; color: #888; margin-bottom: 2rem;">Intelligent Research Assistant • Document Analysis • Real-time Insights</p>', unsafe_allow_html=True)

    # session-state init
    initialize_session_state()

    # Sidebar: validation, system controls, status
    with st.sidebar:
        st.markdown("## 🎛️ Control Panel")

        # API & env validation
        errors = validate_environment()
        if errors:
            for err in errors:
                st.error(f"🔴 {err}")
            # stop further UI; user must fix environment
            return

        # System status cards
        st.markdown("### 📈 System Status")
        s_col1, s_col2 = st.columns(2)
        with s_col1:
            ai_status = "✅" if st.session_state.llm else "❌"
            st.markdown(f"""
            <div class="neuro-card" style="text-align: center;">
                <div>🤖 AI Model</div>
                <div class="stat-number">{ai_status}</div>
            </div>
            """, unsafe_allow_html=True)
        with s_col2:
            docs_status = "✅" if st.session_state.documents_loaded else "⚠️"
            st.markdown(f"""
            <div class="neuro-card" style="text-align: center;">
                <div>📚 Documents</div>
                <div class="stat-number">{docs_status}</div>
            </div>
            """, unsafe_allow_html=True)

        # Initialize / controls
        if not st.session_state.system_initialized:
            if st.button("🚀 Initialize System", use_container_width=True):
                success, message = initialize_system()
                if success:
                    st.success("🟢 " + message)
                    st.rerun()
                else:
                    st.error("🔴 " + message)
        else:
            st.success("🟢 System Ready")

            # Document stats (robust)
            if st.session_state.documents_loaded and st.session_state.vector_store:
                try:
                    # different vector stores expose different APIs; try common ones
                    vs = st.session_state.vector_store
                    doc_count = None
                    # chroma-like
                    try:
                        doc_count = vs._collection.count()
                    except Exception:
                        pass
                    # simple len
                    if doc_count is None:
                        try:
                            doc_count = len(vs)
                        except Exception:
                            pass
                    # fallback
                    doc_count = doc_count if doc_count is not None else "N/A"

                    st.markdown(f"""
                    <div class="neuro-card">
                        <div style="text-align: center;">
                            <div class="stat-number">{doc_count}</div>
                            <div>Knowledge Chunks</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                except Exception:
                    pass

            # Refresh & Clear Chat
            c1, c2 = st.columns(2)
            with c1:
                if st.button("🔄 Refresh", use_container_width=True):
                    st.session_state.vector_store = None
                    st.session_state.documents_loaded = False
                    st.session_state.embeddings = None
                    st.session_state.llm = None
                    st.session_state.system_initialized = False
                    st.rerun()
            with c2:
                if st.button("🗑️ Clear Chat", use_container_width=True):
                    st.session_state.chat_history = []
                    st.rerun()

    # If not initialized, show welcome and stop here
    if not st.session_state.system_initialized:
        display_welcome_screen()
        return

    # Main area: chat and quick questions
    main_col, side_col = st.columns([2, 1])

    with main_col:
        st.markdown("### 💬 Research Chat")

        # If a quick question was set, use it; otherwise allow user input
        if st.session_state.pending_question:
            query = st.session_state.pending_question
            # clear pending to avoid reuse
            st.session_state.pending_question = None
        else:
            query = st.chat_input("Ask your research question...")

        if query and st.session_state.llm:
            # show user message
            with st.chat_message("user"):
                st.markdown(f'<div class="user-message">{query}</div>', unsafe_allow_html=True)

            # generate and show assistant message
            with st.chat_message("assistant"):
                with st.spinner("🔮 Processing your query..."):
                    try:
                        response_tuple = automatic_search(
                            st.session_state.llm,
                            st.session_state.vector_store,
                            query,
                            st.session_state.chat_history,
                        )
                        # allow both tuple and dict style returns
                        if isinstance(response_tuple, tuple) and len(response_tuple) >= 3:
                            response, chat_history, search_metadata = response_tuple[:3]
                        elif isinstance(response_tuple, dict):
                            response = response_tuple.get("response") or response_tuple.get("answer") or ""
                            chat_history = response_tuple.get("chat_history", st.session_state.chat_history)
                            search_metadata = response_tuple.get("metadata", {})
                        else:
                            raise ValueError("automatic_search returned unexpected format")

                        # update session state
                        st.session_state.chat_history = chat_history
                        st.session_state.current_search_mode = search_metadata

                        # display mode info (safely)
                        mode_key = safe_get(search_metadata, "mode", "unknown")
                        mode_info = get_mode_display_info(mode_key)
                        color = mode_info.get("color", "#667eea")
                        st.markdown(f"""
                        <div class="mode-indicator" style="border-left-color: {color};">
                            <strong>{mode_info.get('icon','')} {mode_info.get('name', mode_key)}</strong>
                            <br>
                            <small>{mode_info.get('description','')} • {safe_get(search_metadata,'confidence',0)}% confidence</small>
                        </div>
                        """, unsafe_allow_html=True)

                        # assistant message
                        st.markdown(f'<div class="assistant-message">{response}</div>', unsafe_allow_html=True)

                        # analytics
                        display_search_analytics(search_metadata)

                    except Exception as e:
                        st.error(f"🔴 Error while searching: {str(e)}")

        # Chat history (show last 5)
        if st.session_state.chat_history:
            st.markdown("### 📜 Conversation History")
            for chat in reversed(st.session_state.chat_history[-5:]):
                q = chat.get("question") or chat.get("query") or ""
                a = chat.get("answer") or chat.get("response") or ""
                metadata = chat.get("metadata") or chat.get("meta") or {}
                st.markdown(f'<div class="user-message">{q}</div>', unsafe_allow_html=True)
                if metadata:
                    mode_info = get_mode_display_info(safe_get(metadata, "mode", "unknown"))
                    st.caption(f"{mode_info.get('icon','')} {mode_info.get('name','Unknown')} • {safe_get(metadata,'confidence',0)}% confidence")
                st.markdown(f'<div class="assistant-message">{a}</div>', unsafe_allow_html=True)
                st.markdown("---")

    with side_col:
        display_quick_questions()

if __name__ == "__main__":
    main()
