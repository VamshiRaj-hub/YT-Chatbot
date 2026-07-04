"""
app.py
------
YouTube Video Q&A Chatbot — Main Streamlit Application
"""

import hashlib
import html
import logging
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage

from utils.config import GOOGLE_API_KEY
from utils.config_groq import GROQ_API_KEY
from utils.helpers import is_valid_youtube_url, sanitize_url, extract_video_id
from utils.loader import load_transcript, TranscriptLoadError
from utils.vector_store import (
    split_documents,
    build_vector_store,
    clear_vector_store,
    get_document_count,
)
from utils.rag_chain import (
    build_rag_chain as build_rag_chain_gemini,
    invoke_chain as invoke_chain_gemini,
)
from utils.rag_chain_groq import (
    build_rag_chain_groq,
    invoke_chain_groq,
)

logger = logging.getLogger(__name__)

PROVIDER_LABELS = {
    "groq": "Groq",
    "gemini": "Gemini",
}

PROVIDER_HELP = {
    "groq": "Get a free Groq key at console.groq.com",
    "gemini": "Get a Gemini key from Google AI Studio",
}

PROVIDER_PLACEHOLDERS = {
    "groq": "gsk_••••••••••••••••",
    "gemini": "AIza••••••••••••••••",
}

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="YouTube Q&A Chatbot",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Styles
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Root ── */
html, body, [class*="css"] {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, sans-serif;
}
.main .block-container {
    padding: 2rem 2.5rem 3rem;
    max-width: 820px;
}
#MainMenu, footer, header { visibility: hidden; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #0A0A0F;
    border-right: 1px solid #1E1E2E;
    width: 310px !important;
}
section[data-testid="stSidebar"] .block-container {
    padding: 1.5rem 1.25rem;
}

/* ── Sidebar brand ── */
.brand {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 4px;
}
.brand-icon {
    width: 36px; height: 36px;
    background: linear-gradient(135deg, #FF0000, #FF6B35);
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 18px; line-height: 1;
}
.brand-name {
    font-size: 1.1rem;
    font-weight: 800;
    color: #F9FAFB;
    letter-spacing: -0.3px;
}
.brand-sub {
    font-size: 0.7rem;
    color: #4B5563;
    margin-bottom: 1.25rem;
    padding-left: 2px;
}

/* ── Sidebar section labels ── */
.s-label {
    font-size: 0.68rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #4B5563;
    margin: 1.4rem 0 0.5rem;
}

/* ── Input overrides inside sidebar ── */
section[data-testid="stSidebar"] .stTextInput > label { display: none; }
section[data-testid="stSidebar"] .stTextInput input {
    background: #111118 !important;
    border: 1.5px solid #1E1E2E !important;
    border-radius: 10px !important;
    color: #F9FAFB !important;
    font-size: 0.85rem !important;
    padding: 0.55rem 0.85rem !important;
    transition: border-color 0.2s;
}
section[data-testid="stSidebar"] .stTextInput input:focus {
    border-color: #6366F1 !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.15) !important;
}
section[data-testid="stSidebar"] .stTextInput input::placeholder {
    color: #374151 !important;
}

/* ── Process button ── */
section[data-testid="stSidebar"] .stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #6366F1, #8B5CF6) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    font-size: 0.88rem !important;
    padding: 0.6rem !important;
    width: 100% !important;
    transition: opacity 0.2s, transform 0.1s !important;
    margin-top: 0.75rem;
}
section[data-testid="stSidebar"] .stButton > button[kind="primary"]:hover {
    opacity: 0.9 !important;
    transform: translateY(-1px) !important;
}
section[data-testid="stSidebar"] .stButton > button[kind="primary"]:disabled {
    background: #1E1E2E !important;
    color: #374151 !important;
    cursor: not-allowed !important;
}

/* ── Status pill ── */
.pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 5px 12px;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 600;
    margin-top: 0.6rem;
    width: 100%;
    justify-content: center;
}
.pill-ready   { background: #052e16; color: #4ade80; border: 1px solid #16a34a; }
.pill-idle    { background: #111118; color: #4B5563; border: 1px solid #1E1E2E; }

/* ── Stat cards ── */
.stats-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
    margin: 0.75rem 0;
}
.stat-card {
    background: #111118;
    border: 1px solid #1E1E2E;
    border-radius: 12px;
    padding: 0.7rem 0.5rem;
    text-align: center;
}
.stat-n {
    font-size: 1.5rem;
    font-weight: 800;
    color: #A5B4FC;
    line-height: 1;
}
.stat-t {
    font-size: 0.62rem;
    font-weight: 600;
    color: #4B5563;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 3px;
}
.video-id-badge {
    background: #111118;
    border: 1px solid #1E1E2E;
    border-radius: 8px;
    padding: 6px 10px;
    font-size: 0.72rem;
    color: #818CF8;
    font-family: monospace;
    word-break: break-all;
    margin-top: 4px;
}

/* ── Control buttons ── */
section[data-testid="stSidebar"] .stButton > button:not([kind="primary"]) {
    background: #111118 !important;
    color: #9CA3AF !important;
    border: 1px solid #1E1E2E !important;
    border-radius: 8px !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    transition: background 0.2s, color 0.2s !important;
}
section[data-testid="stSidebar"] .stButton > button:not([kind="primary"]):hover {
    background: #1E1E2E !important;
    color: #F9FAFB !important;
}

/* ── Sidebar divider ── */
section[data-testid="stSidebar"] hr {
    border-color: #1E1E2E !important;
    margin: 1rem 0 !important;
}

/* ── Main page title ── */
.page-title {
    font-size: 1.85rem;
    font-weight: 800;
    color: #111827;
    letter-spacing: -0.5px;
    line-height: 1.2;
    margin-bottom: 4px;
}
.page-subtitle {
    font-size: 0.9rem;
    color: #6B7280;
    margin-bottom: 1.5rem;
}

/* ── Welcome screen ── */
.welcome-wrap {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 3rem 1rem;
}
.welcome-card {
    background: white;
    border: 1px solid #E5E7EB;
    border-radius: 20px;
    padding: 2.5rem 2rem;
    max-width: 480px;
    width: 100%;
    text-align: center;
    box-shadow: 0 4px 24px rgba(0,0,0,0.06);
}
.wc-icon { font-size: 3.5rem; margin-bottom: 1rem; }
.wc-title {
    font-size: 1.25rem;
    font-weight: 700;
    color: #111827;
    margin-bottom: 0.5rem;
}
.wc-desc {
    font-size: 0.875rem;
    color: #6B7280;
    line-height: 1.6;
    margin-bottom: 1.5rem;
}
.wc-steps {
    background: #F9FAFB;
    border-radius: 12px;
    padding: 1rem 1.25rem;
    text-align: left;
}
.wc-step {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 6px 0;
    font-size: 0.85rem;
    color: #374151;
    border-bottom: 1px solid #E5E7EB;
}
.wc-step:last-child { border-bottom: none; }
.wc-step-num {
    width: 24px; height: 24px;
    background: linear-gradient(135deg, #6366F1, #8B5CF6);
    border-radius: 50%;
    color: white;
    font-size: 0.7rem;
    font-weight: 700;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
}

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    padding: 0.25rem 0;
}

/* ── Source chunks ── */
.src-chunk {
    background: #FAFAFA;
    border-left: 3px solid #8B5CF6;
    border-radius: 0 10px 10px 0;
    padding: 0.65rem 0.9rem;
    margin: 0.4rem 0;
    font-size: 0.82rem;
    line-height: 1.65;
    color: #374151;
}
.src-label {
    font-size: 0.65rem;
    font-weight: 700;
    color: #8B5CF6;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 4px;
}

/* ── Chat input ── */
[data-testid="stChatInput"] {
    border-radius: 14px !important;
    border: 1.5px solid #E5E7EB !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06) !important;
}
</style>
""", unsafe_allow_html=True)

# ── JS: auto-commit text inputs on paste (no Enter needed) ──────────────────
st.components.v1.html("""
<script>
(function() {
    function attachPasteListeners() {
        const inputs = window.parent.document.querySelectorAll(
            'section[data-testid="stSidebar"] input[type="password"], ' +
            'section[data-testid="stSidebar"] input[type="text"]'
        );
        inputs.forEach(function(input) {
            if (input._pasteAttached) return;
            input._pasteAttached = true;
            input.addEventListener('paste', function() {
                setTimeout(function() {
                    input.dispatchEvent(new KeyboardEvent('keydown', {
                        key: 'Enter', code: 'Enter', keyCode: 13,
                        bubbles: true, cancelable: true
                    }));
                    input.dispatchEvent(new KeyboardEvent('keyup', {
                        key: 'Enter', code: 'Enter', keyCode: 13,
                        bubbles: true, cancelable: true
                    }));
                }, 100);
            });
        });
    }
    attachPasteListeners();
    setInterval(attachPasteListeners, 1000);
})();
</script>
""", height=0)


# ─────────────────────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────────────────────
def init_session_state():
    defaults = {
        "chat_history": [],
        "lc_chat_history": [],
        "vector_store": None,
        "rag_chain": None,
        "current_video_id": None,
        "selected_provider": "groq",
        "active_provider": None,
        "active_api_key_hash": None,
        "num_chunks": 0,
        "num_docs": 0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session_state()


def _api_key_hash(api_key: str) -> str:
    """Store a fingerprint instead of the raw API key in session state."""
    return hashlib.sha256(api_key.strip().encode("utf-8")).hexdigest()


def _build_provider_chain(provider: str, vector_store, api_key: str):
    if provider == "gemini":
        return build_rag_chain_gemini(vector_store, api_key)
    return build_rag_chain_groq(vector_store, api_key)


def _invoke_provider_chain(provider: str, chain, question: str, chat_history: list):
    if provider == "gemini":
        return invoke_chain_gemini(chain, question, chat_history)
    return invoke_chain_groq(chain, question, chat_history)


def _clear_loaded_video_state():
    st.session_state.vector_store = None
    st.session_state.rag_chain = None
    st.session_state.current_video_id = None
    st.session_state.active_provider = None
    st.session_state.active_api_key_hash = None
    st.session_state.num_chunks = 0
    st.session_state.num_docs = 0
    st.session_state.chat_history = []
    st.session_state.lc_chat_history = []


def _clear_conversation_state():
    st.session_state.chat_history = []
    st.session_state.lc_chat_history = []


def _render_source_chunks(source_docs):
    with st.expander(f"📄 {len(source_docs)} source chunks"):
        for i, doc in enumerate(source_docs, 1):
            safe_content = html.escape(getattr(doc, "page_content", ""))
            st.markdown(
                f'<div class="src-chunk">'
                f'<div class="src-label">Chunk {i}</div>'
                f'{safe_content}'
                f'</div>',
                unsafe_allow_html=True,
            )


# ─────────────────────────────────────────────────────────────────────────────
# Video processing
# ─────────────────────────────────────────────────────────────────────────────
def process_video(url: str, api_key: str, provider: str):
    incoming_id = extract_video_id(sanitize_url(url))
    incoming_key_hash = _api_key_hash(api_key)

    if (
        incoming_id == st.session_state.current_video_id
        and st.session_state.vector_store is not None
        and st.session_state.rag_chain is not None
        and provider == st.session_state.active_provider
        and incoming_key_hash == st.session_state.active_api_key_hash
    ):
        st.sidebar.success(
            f"This video is already loaded with {PROVIDER_LABELS[provider]}."
        )
        return

    with st.sidebar:
        if (
            incoming_id == st.session_state.current_video_id
            and st.session_state.vector_store is not None
        ):
            progress = st.progress(0, text="Refreshing assistant...")
            try:
                new_chain = _build_provider_chain(
                    provider,
                    st.session_state.vector_store,
                    api_key.strip(),
                )
                st.session_state.rag_chain = new_chain
                st.session_state.active_provider = provider
                st.session_state.active_api_key_hash = incoming_key_hash
                _clear_conversation_state()
                progress.progress(100, text="Ready!")
                import time; time.sleep(0.4)
                progress.empty()
                st.rerun()
            except Exception as e:
                st.error(f"❌ RAG chain failed: {e}")
                logger.exception("RAG chain refresh failed")
                progress.empty()
            return

        _clear_loaded_video_state()

        progress = st.progress(0, text="Loading transcript…")
        try:
            docs, video_id = load_transcript(sanitize_url(url))
        except TranscriptLoadError as e:
            st.error(f"❌ {e}")
            progress.empty()
            return

        progress.progress(33, text="Chunking & embedding…")
        chunks = split_documents(docs)
        st.session_state.num_chunks = len(chunks)

        try:
            vs = build_vector_store(chunks, video_id)
            st.session_state.vector_store = vs
            st.session_state.num_docs = get_document_count(vs)
            st.session_state.current_video_id = video_id
        except Exception as e:
            st.error(f"❌ Embedding failed: {e}")
            logger.exception("Vector store build failed")
            progress.empty()
            return

        progress.progress(75, text="Building RAG chain…")
        try:
            st.session_state.rag_chain = _build_provider_chain(
                provider,
                vs,
                api_key.strip(),
            )
            st.session_state.active_provider = provider
            st.session_state.active_api_key_hash = incoming_key_hash
        except Exception as e:
            st.error(f"❌ RAG chain failed: {e}")
            logger.exception("RAG chain failed")
            progress.empty()
            return

        progress.progress(100, text="Ready!")
        import time; time.sleep(0.4)
        progress.empty()
    st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:

    # Brand
    st.markdown("""
    <div class="brand">
        <div class="brand-icon">🎬</div>
        <div class="brand-name">YouTube Q&A</div>
    </div>
    <div class="brand-sub">Groq · LangChain · ChromaDB</div>
    """, unsafe_allow_html=True)

    st.divider()

    # LLM provider
    st.markdown('<div class="s-label">LLM Provider</div>', unsafe_allow_html=True)
    provider_choice = st.selectbox(
        "llm_provider",
        options=["groq", "gemini"],
        format_func=lambda p: PROVIDER_LABELS[p],
        index=0 if st.session_state.selected_provider == "groq" else 1,
        label_visibility="collapsed",
    )
    st.session_state.selected_provider = provider_choice

    # API Key
    default_api_key = GROQ_API_KEY if provider_choice == "groq" else GOOGLE_API_KEY
    if default_api_key.startswith("YOUR_"):
        default_api_key = ""

    st.markdown(
        f'<div class="s-label">{PROVIDER_LABELS[provider_choice]} API Key</div>',
        unsafe_allow_html=True,
    )
    api_key_input = st.text_input(
        "api_key",
        type="password",
        placeholder=PROVIDER_PLACEHOLDERS[provider_choice],
        help=PROVIDER_HELP[provider_choice],
        value=default_api_key,
        key=f"{provider_choice}_api_key_input",
        label_visibility="collapsed",
    )

    # YouTube URL
    st.markdown('<div class="s-label">YouTube URL</div>', unsafe_allow_html=True)
    youtube_url_input = st.text_input(
        "yt_url",
        placeholder="https://youtube.com/watch?v=...",
        label_visibility="collapsed",
    )

    # URL validation hint
    if youtube_url_input.strip() and not is_valid_youtube_url(youtube_url_input.strip()):
        st.caption("⚠️ Not a valid YouTube URL")

    both_ready = bool(api_key_input.strip()) and is_valid_youtube_url(youtube_url_input.strip())

    # Manual process button — only way to trigger processing
    if st.button("⚙️  Process Video", type="primary", use_container_width=True,
                 disabled=not both_ready):
        process_video(youtube_url_input.strip(), api_key_input.strip(), provider_choice)

    st.divider()

    # Status pill
    if st.session_state.rag_chain:
        active_label = PROVIDER_LABELS.get(st.session_state.active_provider, "LLM")
        st.markdown(
            f'<div class="pill pill-ready">Ready with {active_label}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown('<div class="pill pill-idle">○ No video loaded</div>', unsafe_allow_html=True)

    # Stats
    if st.session_state.current_video_id:
        safe_video_id = html.escape(st.session_state.current_video_id)
        st.markdown('<div class="s-label" style="margin-top:1.2rem">Video Stats</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-n">{st.session_state.num_chunks}</div>
                <div class="stat-t">Chunks</div>
            </div>
            <div class="stat-card">
                <div class="stat-n">{st.session_state.num_docs}</div>
                <div class="stat-t">Vectors</div>
            </div>
        </div>
        <div class="video-id-badge">▶ {safe_video_id}</div>
        """, unsafe_allow_html=True)

    st.divider()

    # Controls
    st.markdown('<div class="s-label">Controls</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🗑️ Clear DB", use_container_width=True):
            if clear_vector_store():
                _clear_loaded_video_state()
                st.rerun()
            else:
                st.error("Failed to clear DB.")
    with c2:
        if st.button("💬 Clear Chat", use_container_width=True):
            _clear_conversation_state()
            st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Main content
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<p class="page-title">🎬 YouTube Video Q&A</p>
<p class="page-subtitle">
    Ask any question about a YouTube video and get answers grounded in its transcript.
</p>
""", unsafe_allow_html=True)

st.divider()

# ── No video loaded → welcome screen ──────────────────────────────────────────
if not st.session_state.rag_chain:
    st.markdown("""
    <div class="welcome-wrap">
      <div class="welcome-card">
        <div class="wc-icon">🎬</div>
        <div class="wc-title">Ask anything about any video</div>
        <div class="wc-desc">
          Choose Groq or Gemini, paste the matching API key, and add a YouTube URL.
          Then click Process Video to prepare the chat.
        </div>
        <div class="wc-steps">
          <div class="wc-step">
            <div class="wc-step-num">1</div>
            Choose a provider and enter its <strong>API key</strong>
          </div>
          <div class="wc-step">
            <div class="wc-step-num">2</div>
            Paste a <strong>YouTube URL</strong> with captions
          </div>
          <div class="wc-step">
            <div class="wc-step-num">3</div>
            Click <strong>Process Video</strong>
          </div>
          <div class="wc-step">
            <div class="wc-step-num">4</div>
            Start asking questions ✨
          </div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

# ── Chat interface ─────────────────────────────────────────────────────────────
else:
    # Render history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                with st.expander(f"📄 {len(msg['sources'])} source chunks"):
                    for i, doc in enumerate(msg["sources"], 1):
                        st.markdown(
                            f'<div class="src-chunk">'
                            f'<div class="src-label">Chunk {i}</div>'
                            f'{html.escape(getattr(doc, "page_content", ""))}'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

    # Input
    if question := st.chat_input("Ask a question about the video…"):
        question = question.strip()
        if not question:
            st.stop()
        st.session_state.chat_history.append({"role": "user", "content": question})

        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner(""):
                try:
                    result = _invoke_provider_chain(
                        st.session_state.active_provider,
                        st.session_state.rag_chain,
                        question,
                        st.session_state.lc_chat_history,
                    )
                    answer = result.get("answer", "")
                    source_docs = result.get("context", [])
                except Exception as e:
                    logger.exception("RAG chain error")
                    answer = f"❌ An error occurred: {e}"
                    source_docs = []

            st.write(answer)

            if source_docs:
                with st.expander(f"📄 {len(source_docs)} source chunks"):
                    for i, doc in enumerate(source_docs, 1):
                        st.markdown(
                            f'<div class="src-chunk">'
                            f'<div class="src-label">Chunk {i}</div>'
                            f'{html.escape(getattr(doc, "page_content", ""))}'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer,
            "sources": source_docs,
        })
        st.session_state.lc_chat_history.append(HumanMessage(content=question))
        st.session_state.lc_chat_history.append(AIMessage(content=answer))
