"""
LexBridge — Streamlit frontend for the RAG API.
Run: streamlit run frontend/app.py
"""

import os
import streamlit as st
import requests
import time

API_BASE = os.environ.get("API_BASE", "http://localhost:8000")

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LexBridge — U.S. Employment Law RAG",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
    /* Hide default Streamlit chrome */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Animated gradient background */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    }

    /* Hero header */
    .hero {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.15) 0%, rgba(118, 75, 162, 0.15) 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 2.5rem 2rem;
        margin-bottom: 2rem;
        backdrop-filter: blur(10px);
        text-align: center;
    }
    .hero h1 {
        font-size: 3.5rem;
        font-weight: 800;
        margin: 0;
        background: linear-gradient(135deg, #667eea 0%, #f093fb 50%, #4facfe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: -2px;
    }
    .hero p {
        color: rgba(255, 255, 255, 0.7);
        font-size: 1.1rem;
        margin-top: 0.5rem;
        font-weight: 300;
    }

    /* Answer card */
    .answer-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-left: 4px solid #667eea;
        border-radius: 16px;
        padding: 2rem;
        margin: 1.5rem 0;
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    .answer-card .label {
        color: #a78bfa;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 1rem;
    }
    .answer-card .answer-text {
        color: #f0f0f0;
        font-size: 1.1rem;
        line-height: 1.8;
        font-weight: 400;
    }

    /* Source card */
    .source-card {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin: 0.8rem 0;
        transition: all 0.3s ease;
    }
    .source-card:hover {
        background: rgba(255, 255, 255, 0.07);
        border-color: rgba(102, 126, 234, 0.3);
        transform: translateX(4px);
    }
    .source-meta {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.8rem;
        flex-wrap: wrap;
        gap: 0.5rem;
    }
    .source-topic {
        color: #4facfe;
        font-weight: 600;
        font-size: 0.95rem;
    }
    .source-name {
        color: rgba(255, 255, 255, 0.5);
        font-size: 0.85rem;
    }
    .source-text {
        color: rgba(255, 255, 255, 0.75);
        font-size: 0.9rem;
        line-height: 1.6;
        font-style: italic;
    }

    /* Score badge */
    .score-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
    }
    .score-high { background: rgba(34, 197, 94, 0.2); color: #4ade80; border: 1px solid rgba(34, 197, 94, 0.3); }
    .score-mid  { background: rgba(251, 191, 36, 0.2); color: #fbbf24; border: 1px solid rgba(251, 191, 36, 0.3); }
    .score-low  { background: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.3); }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 1.5rem;
        font-weight: 600;
        font-size: 0.95rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }

    /* Suggestion chips */
    .suggestion-row { display: flex; gap: 0.5rem; flex-wrap: wrap; margin: 1rem 0; }

    /* Health pill */
    .health-pill {
        display: flex; align-items: center; gap: 0.6rem;
        padding: 0.7rem 1rem;
        border-radius: 10px;
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        margin-bottom: 1rem;
        font-size: 0.9rem;
    }
    .dot { width: 10px; height: 10px; border-radius: 50%; }
    .dot-green { background: #4ade80; box-shadow: 0 0 12px #4ade80; }
    .dot-red   { background: #f87171; box-shadow: 0 0 12px #f87171; }

    /* Inputs */
    .stTextInput > div > div > input,
    .stTextArea textarea {
        background: rgba(255, 255, 255, 0.05) !important;
        color: white !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        border-radius: 10px !important;
    }
    .stTextInput > div > div > input:focus,
    .stTextArea textarea:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.2) !important;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: rgba(15, 12, 41, 0.6);
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }

    /* Stats row */
    .stat-card {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
    }
    .stat-value {
        font-size: 1.6rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .stat-label {
        color: rgba(255, 255, 255, 0.6);
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 0.3rem;
    }
</style>
""",
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def health_check():
    try:
        r = requests.get(f"{API_BASE}/", timeout=3)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


def ask(text, k, language):
    r = requests.post(
        f"{API_BASE}/api/v1/nlp/ask",
        json={"text": text, "collection": "lexbridge", "k": k, "language": language},
        timeout=60,
    )
    r.raise_for_status()
    return r.json()


def upload(file):
    files = {"file": (file.name, file.getvalue(), file.type)}
    data = {"collection": "lexbridge"}
    r = requests.post(
        f"{API_BASE}/api/v1/data/upload", files=files, data=data, timeout=120
    )
    r.raise_for_status()
    return r.json()


def score_class(score):
    if score >= 0.7:
        return "score-high"
    if score >= 0.5:
        return "score-mid"
    return "score-low"


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚖️ LexBridge")
    st.caption("U.S. Employment Law RAG")
    st.divider()

    # Health status
    health = health_check()
    if health:
        st.markdown(
            f'<div class="health-pill">'
            f'<span class="dot dot-green"></span>'
            f"<div><b>API online</b><br>"
            f'<span style="opacity:0.6;font-size:0.8rem">DB: {"✓" if health["db_connected"] else "✗"} · '
            f'Vector: {"✓" if health["vector_db_ready"] else "✗"}</span></div>'
            f"</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="health-pill">'
            '<span class="dot dot-red"></span>'
            "<div><b>API offline</b><br>"
            '<span style="opacity:0.6;font-size:0.8rem">Start docker-compose</span></div>'
            "</div>",
            unsafe_allow_html=True,
        )

    st.markdown("##### Settings")
    language = st.selectbox(
        "Response language",
        options=["en", "ar"],
        format_func=lambda x: "🇺🇸 English" if x == "en" else "🇸🇦 العربية",
    )
    k = st.slider("Sources retrieved", 1, 10, 5)

    st.divider()
    st.markdown("##### Upload document")
    uploaded = st.file_uploader(
        "PDF or TXT", type=["pdf", "txt"], label_visibility="collapsed"
    )
    if uploaded and st.button("Index document", use_container_width=True):
        with st.spinner("Embedding & indexing..."):
            try:
                result = upload(uploaded)
                st.success(f"✓ {result['chunks_created']} chunks added")
            except Exception as e:
                st.error(f"Upload failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Main content
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
<div class="hero">
    <h1>LexBridge</h1>
    <p>U.S. Employment Law · Powered by RAG · 34 sources, 446 indexed chunks</p>
</div>
""",
    unsafe_allow_html=True,
)

EN_SUGGESTIONS = [
    "What is at-will employment?",
    "What does the FMLA require?",
    "What is the federal minimum wage?",
    "What is wrongful termination?",
    "What are whistleblower protections?",
]

AR_SUGGESTIONS = [
    "ما هو التوظيف الاختياري (At-Will)؟",
    "ما الذي يتطلبه قانون إجازة الأسرة والطب (FMLA)؟",
    "ما هو الحد الأدنى الفيدرالي للأجور؟",
    "ما هو الفصل التعسفي من العمل؟",
    "ما هي حمايات المُبلِّغين عن المخالفات؟",
]

SUGGESTIONS = AR_SUGGESTIONS if language == "ar" else EN_SUGGESTIONS

if "query" not in st.session_state:
    st.session_state.query = ""

st.markdown("##### 💡 Try asking")
cols = st.columns(len(SUGGESTIONS))
for i, s in enumerate(SUGGESTIONS):
    with cols[i]:
        if st.button(s, key=f"sug_{i}", use_container_width=True):
            st.session_state.query = s

# Query input
query = st.text_area(
    "Question",
    value=st.session_state.query,
    placeholder="Ask anything about U.S. employment law...",
    height=100,
    label_visibility="collapsed",
)

col1, col2 = st.columns([1, 5])
with col1:
    submit = st.button("🔍 Ask", use_container_width=True, type="primary")

# Results
if submit and query.strip():
    start_time = time.time()
    try:
        with st.spinner("🧠 Thinking..."):
            result = ask(query.strip(), k, language)
        elapsed = time.time() - start_time

        # Stats row
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(
                f'<div class="stat-card"><div class="stat-value">{len(result["sources"])}</div><div class="stat-label">Sources</div></div>',
                unsafe_allow_html=True,
            )
        with c2:
            top_score = result["sources"][0]["score"] if result["sources"] else 0
            st.markdown(
                f'<div class="stat-card"><div class="stat-value">{top_score:.2f}</div><div class="stat-label">Top Score</div></div>',
                unsafe_allow_html=True,
            )
        with c3:
            st.markdown(
                f'<div class="stat-card"><div class="stat-value">{elapsed:.1f}s</div><div class="stat-label">Latency</div></div>',
                unsafe_allow_html=True,
            )
        with c4:
            model_short = result["model"].split("/")[-1].replace("gemini-", "")
            st.markdown(
                f'<div class="stat-card"><div class="stat-value">{model_short}</div><div class="stat-label">Model</div></div>',
                unsafe_allow_html=True,
            )

        # Answer
        direction = "rtl" if language == "ar" else "ltr"
        st.markdown(
            f'<div class="answer-card">'
            f'<div class="label">📝 Answer</div>'
            f'<div class="answer-text" dir="{direction}">{result["answer"]}</div>'
            f"</div>",
            unsafe_allow_html=True,
        )

        # Sources & Chunks
        with st.expander(
            f"📚 View {len(result['sources'])} retrieved sources", expanded=False
        ):
            for i, src in enumerate(result["sources"], 1):
                cls = score_class(src["score"])
                url_link = (
                    f'<a href="{src["source_url"]}" style="color:#4facfe;text-decoration:none;font-weight:600">→ View document</a>'
                    if src.get("source_url")
                    else ""
                )

                st.markdown(
                    f'<div class="source-card">'
                    f'<div class="source-meta">'
                    f'<span class="source-topic">#{i} · {src["topic"]}</span>'
                    f'<span><span class="score-badge {cls}">{src["score"]:.3f}</span></span>'
                    f"</div>"
                    f'<div style="margin-top:0.8rem;margin-bottom:0.8rem"><b>Source:</b> <span class="source-name">{src["source_name"]}</span></div>'
                    f"</div>",
                    unsafe_allow_html=True,
                )

                with st.container(border=False):
                    st.markdown("**Chunk text:**")
                    st.markdown(
                        f'<div style="background:rgba(255,255,255,0.02);border-left:3px solid #667eea;padding:1rem;border-radius:8px;color:rgba(255,255,255,0.85);font-size:0.95rem;line-height:1.7">{src["text"]}</div>',
                        unsafe_allow_html=True,
                    )

                    if url_link:
                        st.markdown(
                            f'<div style="margin-top:0.8rem">{url_link}</div>',
                            unsafe_allow_html=True,
                        )

    except requests.HTTPError as e:
        st.error(f"API error: {e.response.status_code} — {e.response.text[:200]}")
    except Exception as e:
        st.error(f"Error: {e}")

elif submit:
    st.warning("Type a question first.")
