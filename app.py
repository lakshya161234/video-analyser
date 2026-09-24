import streamlit as st
import html
import os
from dotenv import load_dotenv

load_dotenv()

from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarizer import summarize, generate_title
from core.extractor import extract_meeting_insights
from core.rag_engine import build_rag_chain, ask_question


def display_error(error: Exception) -> str:
    response = getattr(error, "response", None)
    status_code = getattr(response, "status_code", getattr(error, "code", None))
    provider = os.getenv("LLM_PROVIDER", "mistral").strip().lower()
    provider_names = {"gemini": "Gemini", "groq": "Groq", "mistral": "Mistral"}
    provider_name = provider_names.get(provider, "LLM provider")
    if status_code == 429:
        details = None
        retry_after = None
        if response is not None:
            retry_after = getattr(response, "headers", {}).get("retry-after")
            try:
                payload = response.json()
                provider_error = payload.get("error", {})
                if isinstance(provider_error, dict):
                    details = provider_error.get("message") or provider_error.get("code")
                elif provider_error:
                    details = str(provider_error)
            except Exception:
                details = None
        message = (
            f"{provider_name} API rate limit reached (HTTP 429). Check your API "
            "usage and limits, then wait and try again."
        )
        if details:
            message += f" Provider detail: {details}"
        if retry_after:
            message += f" Retry after {retry_after} seconds."
        return message + " If you changed the key, update .env and restart the app."
    if status_code in (401, 403):
        key_names = {
            "gemini": "GOOGLE_API_KEY",
            "groq": "GROQ_API_KEY",
            "mistral": "MISTRAL_API_KEY",
        }
        key_name = key_names.get(provider, "API key")
        return f"{provider_name} rejected the API key. Check {key_name} in .env and restart the app."
    return f"{type(error).__name__}: {error}"

                                                                                  
st.set_page_config(
    page_title="AI Video Assistant",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

                                                                                  
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');

:root {
    --page: #f4f6f8;
    --surface: #ffffff;
    --surface-muted: #f8fafb;
    --line: #e2e8ed;
    --ink: #18232e;
    --muted: #667582;
    --accent: #176b63;
    --accent-hover: #11574f;
    --accent-soft: #e8f3f1;
    --success: #287a58;
    --warning: #9a6516;
    --danger: #a13e45;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background: var(--page) !important;
    color: var(--ink) !important;
}

.stApp { background: var(--page) !important; }
[data-testid="stAppViewContainer"] { background: var(--page) !important; }
[data-testid="stHeader"] { background: rgba(244,246,248,.92) !important; }
[data-testid="stMainBlockContainer"] {
    max-width: 1440px;
    padding: 2.1rem 3rem 4rem;
}

[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--line) !important;
}
[data-testid="stSidebar"] > div:first-child { background: var(--surface) !important; }
[data-testid="stSidebarContent"] { padding: 1.6rem 1.2rem 2rem !important; }
[data-testid="stSidebar"] * { color: var(--ink); }

h1, h2, h3, h4 {
    color: var(--ink) !important;
    font-family: 'Manrope', 'DM Sans', sans-serif !important;
    letter-spacing: -0.035em;
}
h1 { font-size: clamp(2rem, 3.4vw, 3rem) !important; line-height: 1.12 !important; }
h2 { font-size: 1.45rem !important; line-height: 1.25 !important; }
h3 { font-size: 1.05rem !important; line-height: 1.35 !important; }
p, li { line-height: 1.7; }

.eyebrow {
    color: var(--accent);
    font-size: .72rem;
    font-weight: 700;
    letter-spacing: .13em;
    text-transform: uppercase;
    margin: 0 0 .55rem;
}
.page-subtitle { color: var(--muted); font-size: 1rem; margin-top: -.35rem; }
.sidebar-brand {
    font-family: 'Manrope', sans-serif;
    font-size: 1.05rem;
    font-weight: 800;
    letter-spacing: -.025em;
    color: var(--ink);
}
.sidebar-note { color: var(--muted); font-size: .83rem; line-height: 1.55; }
.section-label {
    color: var(--muted);
    font-size: .7rem;
    font-weight: 700;
    letter-spacing: .1em;
    text-transform: uppercase;
    margin: .3rem 0 .65rem;
}

.stTextInput input, .stSelectbox [data-baseweb="select"] > div {
    background: var(--surface) !important;
    border: 1px solid #cfd8df !important;
    border-radius: 8px !important;
    color: var(--ink) !important;
    min-height: 2.75rem;
    box-shadow: none !important;
}
.stTextInput input:focus, .stSelectbox [data-baseweb="select"] > div:focus-within {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(23,107,99,.12) !important;
}
[data-testid="stWidgetLabel"] p, label { color: #465562 !important; font-weight: 600 !important; }
.stButton > button {
    min-height: 2.8rem;
    border-radius: 8px !important;
    background: var(--accent) !important;
    color: #fff !important;
    border: 1px solid var(--accent) !important;
    font-weight: 700 !important;
    letter-spacing: .01em;
    box-shadow: none !important;
    transition: background .15s ease, border-color .15s ease;
}
.stButton > button:hover { background: var(--accent-hover) !important; border-color: var(--accent-hover) !important; }
.stButton > button:focus { box-shadow: 0 0 0 3px rgba(23,107,99,.18) !important; }

.content-card {
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: 12px;
    padding: 1.35rem 1.5rem;
    margin: 0 0 1rem;
    min-width: 0;
    box-shadow: 0 2px 8px rgba(24,35,46,.035);
}
.content-card h3 {
    color: #43515d !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: .78rem !important;
    font-weight: 700 !important;
    letter-spacing: .075em;
    margin: 0 0 .75rem !important;
    text-transform: uppercase;
}
.content-card-body {
    color: var(--ink);
    font-size: .94rem;
    line-height: 1.72;
    white-space: pre-wrap;
    overflow-wrap: anywhere;
    word-break: normal;
}
.summary-card { min-height: 100%; }
.transcript-box {
    color: #354451;
    font-size: .88rem;
    line-height: 1.7;
    white-space: pre-wrap;
    overflow-wrap: anywhere;
    max-height: 420px;
    overflow-y: auto;
}

.pipeline-heading { margin: 1.35rem 0 .55rem; }
.pipeline-row {
    display: flex;
    align-items: center;
    gap: .65rem;
    min-height: 2.45rem;
    padding: .36rem .1rem;
    border-bottom: 1px solid #edf0f2;
    color: #43515d;
    font-size: .86rem;
}
.pipeline-row:last-child { border-bottom: 0; }
.pipeline-dot {
    width: .55rem;
    height: .55rem;
    border-radius: 50%;
    flex: 0 0 .55rem;
    background: #c5ced5;
}
.pipeline-row.active { color: var(--ink); font-weight: 700; }
.pipeline-row.active .pipeline-dot { background: #bd8126; box-shadow: 0 0 0 3px #f7efdf; }
.pipeline-row.done .pipeline-dot { background: var(--success); }
.pipeline-row.error { color: var(--danger); font-weight: 700; }
.pipeline-row.error .pipeline-dot { background: var(--danger); }
.pipeline-state { margin-left: auto; color: var(--muted); font-size: .72rem; font-weight: 500; }
.pipeline-row.active .pipeline-state { color: var(--warning); }
.pipeline-row.done .pipeline-state { color: var(--success); }
.pipeline-row.error .pipeline-state { color: var(--danger); }

.empty-state {
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: 14px;
    padding: clamp(2rem, 5vw, 4.5rem) 2rem;
    text-align: center;
    box-shadow: 0 2px 8px rgba(24,35,46,.035);
}
.empty-mark {
    width: 3rem;
    height: 3rem;
    margin: 0 auto 1.1rem;
    border: 1px solid #cfe2de;
    border-radius: 12px;
    background: var(--accent-soft);
    color: var(--accent);
    display: grid;
    place-items: center;
    font-family: 'Manrope', sans-serif;
    font-size: 1.1rem;
    font-weight: 800;
}
.empty-state p { color: var(--muted); margin: .35rem auto 0; max-width: 35rem; }

.chat-container {
    display: flex;
    flex-direction: column;
    gap: .75rem;
    margin: .5rem 0 1rem;
    min-width: 0;
}
.chat-msg { display: flex; flex-direction: column; gap: .28rem; min-width: 0; }
.chat-label { color: var(--muted); font-size: .69rem; font-weight: 700; letter-spacing: .07em; text-transform: uppercase; }
.chat-bubble {
    display: inline-block;
    width: fit-content;
    max-width: min(90%, 52rem);
    padding: .75rem .95rem;
    border: 1px solid var(--line);
    border-radius: 10px;
    color: var(--ink);
    font-size: .92rem;
    line-height: 1.65;
    white-space: pre-wrap;
    overflow-wrap: anywhere;
    background: var(--surface);
}
.user-bubble { align-self: flex-end; background: var(--accent-soft); border-color: #cfe2de; }
.bot-bubble { align-self: flex-start; }

[data-testid="stExpander"] {
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: 10px;
}
[data-testid="stExpander"] summary { color: var(--ink) !important; font-weight: 600; }
[data-testid="stAlert"] { border-radius: 9px; }
[data-testid="stMarkdownContainer"] { min-width: 0; }
[data-testid="stMarkdownContainer"] p { overflow-wrap: anywhere; }
[data-testid="stHorizontalBlock"] { align-items: stretch; gap: 1rem; }
hr { border-color: var(--line) !important; margin: 1.6rem 0 !important; }

@media (max-width: 850px) {
    [data-testid="stMainBlockContainer"] { padding: 1.35rem 1.1rem 3rem; }
    .content-card { padding: 1.1rem; }
    .chat-bubble { max-width: 100%; }
}
</style>
""", unsafe_allow_html=True)

                                                                                   
for key, default in {
    "result": None,
    "chat_history": [],
    "processing": False,
    "pipeline_done": False,
    "pipeline_steps": {},
    "active_pipeline_step": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

                                                                                  
PIPELINE_STEPS = [
    ("audio", "Prepare audio"),
    ("transcript", "Transcribe recording"),
    ("title", "Create meeting title"),
    ("summary", "Write summary"),
    ("extract", "Extract meeting notes"),
    ("rag", "Prepare meeting chat"),
]


def render_pipeline_status(placeholder, steps: dict):
    state_labels = {
        "pending": "Waiting",
        "active": "In progress",
        "done": "Complete",
        "error": "Failed",
    }
    rows = []
    for key, label in PIPELINE_STEPS:
        state = steps.get(key, "pending")
        rows.append(
            f'<div class="pipeline-row {state}">'
            f'<span class="pipeline-dot"></span>'
            f'<span>{label}</span>'
            f'<span class="pipeline-state">{state_labels.get(state, "Waiting")}</span>'
            "</div>"
        )
    placeholder.markdown(
        '<div class="section-label pipeline-heading">Pipeline status</div>'
        + "".join(rows),
        unsafe_allow_html=True,
    )


def render_content_card(title: str, content: str, class_name: str = ""):
    safe_title = html.escape(str(title))
    safe_content = html.escape(str(content))
    st.markdown(
        f'<section class="content-card {class_name}">'
        f"<h3>{safe_title}</h3>"
        f'<div class="content-card-body">{safe_content}</div>'
        "</section>",
        unsafe_allow_html=True,
    )

                                                                                  
with st.sidebar:
    st.markdown('<div class="sidebar-brand">Meeting intelligence</div>', unsafe_allow_html=True)
    st.markdown('<p class="sidebar-note">Turn a recording into clear notes and decisions.</p>', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Recording</div>', unsafe_allow_html=True)
    source = st.text_input(
        "Video or audio source",
        placeholder="Paste a YouTube link or local file path",
        label_visibility="collapsed",
    )
    language = st.selectbox("Transcript language", ["english", "hinglish"], index=0)
    run_btn = st.button("Analyze meeting", type="primary", use_container_width=True)
    st.markdown('<p class="sidebar-note">YouTube links and local media files are supported.</p>', unsafe_allow_html=True)
    pipeline_status_placeholder = st.empty()

render_pipeline_status(pipeline_status_placeholder, st.session_state.pipeline_steps)

st.markdown('<div class="eyebrow">MEETING WORKSPACE</div>', unsafe_allow_html=True)
st.title("AI Video Assistant")
st.markdown('<div class="page-subtitle">Transcripts, concise summaries, decisions, and follow-up—all in one place.</div>', unsafe_allow_html=True)
st.markdown("---")

                                                                                  
if run_btn:
    if not source.strip():
        st.error("Please enter a YouTube URL or file path.")
    else:
        st.session_state.pipeline_done = False
        st.session_state.processing = True
        st.session_state.result = None
        st.session_state.chat_history = []
        st.session_state.pipeline_steps = {key: "pending" for key, _ in PIPELINE_STEPS}
        st.session_state.active_pipeline_step = None

        progress_placeholder = st.empty()

        def update_step(key, state):
            st.session_state.pipeline_steps[key] = state
            st.session_state.active_pipeline_step = key if state == "active" else None
            render_pipeline_status(pipeline_status_placeholder, st.session_state.pipeline_steps)

        try:
            progress_placeholder.info("Analysis is running. You can follow each stage in the sidebar.")

            update_step("audio", "active")
            chunks = process_input(source)
            update_step("audio", "done")

            update_step("transcript", "active")
            transcript = transcribe_all(chunks, language)
            update_step("transcript", "done")

            update_step("title", "active")
            title = generate_title(transcript)
            update_step("title", "done")

            update_step("summary", "active")
            summary = summarize(transcript)
            update_step("summary", "done")

            update_step("extract", "active")
            insights = extract_meeting_insights(transcript)
            action_items = insights["action_items"]
            decisions = insights["key_decisions"]
            questions = insights["open_questions"]
            update_step("extract", "done")

            update_step("rag", "active")
            rag_chain = build_rag_chain(transcript)
            update_step("rag", "done")

            st.session_state.result = {
                "title": title,
                "transcript": transcript,
                "summary": summary,
                "action_items": action_items,
                "key_decisions": decisions,
                "open_questions": questions,
                "rag_chain": rag_chain,
            }
            st.session_state.pipeline_done = True
            st.session_state.processing = False
            progress_placeholder.success("Analysis complete.")
            st.rerun()

        except Exception as e:
            failed_step = st.session_state.active_pipeline_step
            if failed_step:
                update_step(failed_step, "error")
            st.session_state.processing = False
            progress_placeholder.error(display_error(e))

                                                                                   
if st.session_state.result:
    r = st.session_state.result

    render_content_card("Meeting title", r["title"])

    col1, col2 = st.columns([1.4, 1], gap="large")

    with col1:
        render_content_card("Summary", r["summary"], "summary-card")

    with col2:
        with st.expander("View full transcript", expanded=False):
            st.markdown(
                f'<div class="transcript-box">{html.escape(str(r["transcript"]))}</div>',
                unsafe_allow_html=True,
            )

    c1, c2, c3 = st.columns(3, gap="medium")

    with c1:
        render_content_card("Action items", r["action_items"])

    with c2:
        render_content_card("Key decisions", r["key_decisions"])

    with c3:
        render_content_card("Open questions", r["open_questions"])

    st.markdown("---")

    st.subheader("Ask about this meeting")
    st.markdown('<p class="sidebar-note">Answers are grounded in the transcript.</p>', unsafe_allow_html=True)

    if st.session_state.chat_history:
        chat_html = '<div class="chat-container">'
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                label = "You"
                bubble_class = "user-bubble"
            else:
                label = "Assistant"
                bubble_class = "bot-bubble"
            chat_html += f"""
                <div class="chat-msg">
                    <span class="chat-label">{label}</span>
                    <div class="chat-bubble {bubble_class}">{html.escape(str(msg['content']))}</div>
                </div>"""
        chat_html += '</div>'
        st.markdown(chat_html, unsafe_allow_html=True)
    else:
        st.markdown(
            '<div class="content-card"><div class="content-card-body">'
            'Ask a question about a decision, owner, deadline, or topic in the transcript.'
            '</div></div>',
            unsafe_allow_html=True,
        )

    chat_col1, chat_col2 = st.columns([5, 1], gap="small")
    with chat_col1:
        user_input = st.text_input(
            "Your question",
            placeholder="Ask a question about the meeting",
            label_visibility="collapsed",
        )
    with chat_col2:
        send_btn = st.button("Send", use_container_width=True)

    if send_btn and user_input.strip():
        try:
            with st.spinner("Thinking…"):
                answer = ask_question(r["rag_chain"], user_input.strip())
            st.session_state.chat_history.append({"role": "user", "content": user_input.strip()})
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
            st.rerun()
        except Exception as error:
            st.error(display_error(error))

    if st.session_state.chat_history:
        if st.button("Clear conversation", type="secondary"):
            st.session_state.chat_history = []
            st.rerun()

else:
    st.markdown("""
    <div class="empty-state">
        <div class="empty-mark">MI</div>
        <div class="eyebrow">MEETING NOTES, MADE CLEAR</div>
        <h2>Start with a recording</h2>
        <p>Add a YouTube link or local media file in the sidebar. The assistant will create a transcript, summary, action items, and a searchable meeting chat.</p>
    </div>""", unsafe_allow_html=True)
