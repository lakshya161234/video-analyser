import streamlit as st
import html
import os
import base64
import re
import tempfile
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="ClipMind — Meeting Intelligence",
    page_icon=str(Path(__file__).parent / "assets" / "clipmind-mark.svg"),
    layout="wide",
    initial_sidebar_state="expanded",
)

# Streamlit Community Cloud exposes secrets through st.secrets; the application
# modules read provider credentials from environment variables.
for _secret_name in (
    "LLM_PROVIDER",
    "GROQ_API_KEY",
    "MISTRAL_API_KEY",
    "GOOGLE_API_KEY",
    "GEMINI_API_KEY",
    "SARVAM_API_KEY",
    "GROQ_MODEL",
    "MISTRAL_MODEL",
    "GEMINI_MODEL",
    "WHISPER_MODEL",
):
    if not os.getenv(_secret_name):
        try:
            _secret_value = st.secrets.get(_secret_name)
        except Exception:
            _secret_value = None
        if _secret_value:
            os.environ[_secret_name] = str(_secret_value)

from utils.audio_processor import cleanup_input_chunks, process_input
from core.transcriber import transcribe_all
from core.summarizer import summarize, generate_title
from core.extractor import extract_meeting_insights
from core.rag_engine import build_rag_chain, ask_question


def display_error(error: Exception) -> str:
    error_text = str(error)
    if "http error 403: forbidden" in error_text.lower() and "download" in error_text.lower():
        return (
            "YouTube refused the media download (HTTP 403). This can happen when its "
            "network restrictions block the app's cloud host. Upload the audio/video "
            "file instead, or run ClipMind locally to download from your own network."
        )

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
        return message + " If you changed the key, update app secrets or environment settings and restart the app."
    if status_code in (401, 403):
        key_names = {
            "gemini": "GOOGLE_API_KEY",
            "groq": "GROQ_API_KEY",
            "mistral": "MISTRAL_API_KEY",
        }
        key_name = key_names.get(provider, "API key")
        return f"{provider_name} rejected the API key. Check {key_name} in app secrets or environment settings, then restart the app."
    return f"{type(error).__name__}: {error_text}"

                                                                                  

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');

:root {
    color-scheme: dark;
    --page: #081321;
    --surface: #101d30;
    --surface-muted: #14243a;
    --line: #273b55;
    --ink: #edf5ff;
    --muted: #9aadc4;
    --accent: #55dfc2;
    --accent-hover: #77ecd2;
    --accent-soft: #123d3b;
    --button-ink: #062522;
    --blue-soft: #182e4b;
    --blue-ink: #a6c8ff;
    --amber-soft: #3b2d1c;
    --amber-ink: #ffd18a;
    --violet-soft: #302447;
    --violet-ink: #d1b7ff;
    --rose-soft: #3c252d;
    --rose-ink: #ffb5bd;
    --success: #5cdda3;
    --warning: #ffd079;
    --danger: #ff8996;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background: var(--page) !important;
    color: var(--ink) !important;
}

.stApp,
[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(900px 580px at 92% -12%, rgba(42, 106, 168, .22), transparent 62%),
        radial-gradient(720px 540px at -8% 42%, rgba(24, 151, 126, .14), transparent 64%),
        var(--page) !important;
    background-attachment: fixed !important;
}
[data-testid="stHeader"] { background: rgba(8,19,33,.88) !important; }
[data-testid="stMainBlockContainer"] {
    max-width: 1440px;
    padding: 1.25rem 3rem 4rem;
}

[data-testid="stSidebar"] {
    background: #0c192a !important;
    border-right: 1px solid var(--line) !important;
}
section[data-testid="stSidebar"] {
    width: 23rem !important;
    min-width: 23rem !important;
}
[data-testid="stSidebar"] > div:first-child { background: #0c192a !important; }
[data-testid="stSidebarContent"] { padding: 1.6rem 1.2rem 2rem !important; }
[data-testid="stSidebar"] * { color: var(--ink); }

h1, h2, h3, h4 {
    color: var(--ink) !important;
    font-family: 'Manrope', 'DM Sans', sans-serif !important;
    letter-spacing: -0.035em;
}
h1 { font-size: clamp(1.8rem, 2.8vw, 2.35rem) !important; line-height: 1.08 !important; }
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
.main-brand { display: flex; align-items: center; gap: .8rem; margin: .1rem 0 .45rem; }
.main-brand img { width: 2.65rem; height: 2.65rem; flex: 0 0 2.65rem; }
.main-brand h1 { color: #d9fff5 !important; font-size: clamp(1.8rem, 2.8vw, 2.35rem) !important; margin: 0 !important; }
.sidebar-brand {
    display: flex;
    align-items: center;
    gap: .65rem;
    font-family: 'Manrope', sans-serif;
    font-size: 1.12rem;
    font-weight: 800;
    letter-spacing: -.025em;
    color: var(--ink);
}
.brand-logo { width: 2.35rem; height: 2.35rem; flex: 0 0 2.35rem; }
.sidebar-note { color: var(--muted); font-size: .83rem; line-height: 1.55; }
.youtube-note {
    margin: .5rem 0 .7rem;
    padding: .7rem .8rem;
    border-left: 3px solid var(--accent);
    border-radius: 0 8px 8px 0;
    background: var(--surface-muted);
    color: var(--muted);
    font-size: .78rem;
    line-height: 1.5;
}
.youtube-note strong { display: block; margin-bottom: .2rem; color: var(--ink); }
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
    border: 1px solid #304660 !important;
    border-radius: 8px !important;
    color: var(--ink) !important;
    min-height: 2.75rem;
    box-shadow: none !important;
}
.stTextInput input:focus, .stSelectbox [data-baseweb="select"] > div:focus-within {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(85,223,194,.16) !important;
}
[data-testid="stWidgetLabel"] p, label { color: #c0cee0 !important; font-weight: 600 !important; }
.stButton > button, [data-testid="stFormSubmitButton"] > button {
    min-height: 2.8rem;
    border-radius: 8px !important;
    background: var(--accent) !important;
    color: var(--button-ink) !important;
    -webkit-text-fill-color: var(--button-ink) !important;
    border: 1px solid var(--accent) !important;
    font-weight: 700 !important;
    letter-spacing: .01em;
    box-shadow: none !important;
    transition: background .15s ease, border-color .15s ease;
}
.stButton > button *, [data-testid="stFormSubmitButton"] > button * {
    color: var(--button-ink) !important;
    -webkit-text-fill-color: var(--button-ink) !important;
}
.stButton > button:hover, [data-testid="stFormSubmitButton"] > button:hover { background: var(--accent-hover) !important; border-color: var(--accent-hover) !important; }
.stButton > button:focus, [data-testid="stFormSubmitButton"] > button:focus { box-shadow: 0 0 0 3px rgba(85,223,194,.22) !important; }
.stButton > button[kind="secondary"] { background: #192940 !important; color: var(--ink) !important; border-color: var(--line) !important; }
.stButton > button[kind="secondary"] *, .stButton > button[kind="secondary"] p { color: var(--ink) !important; -webkit-text-fill-color: var(--ink) !important; }

[data-testid="stVerticalBlockBorderWrapper"] {
    background: linear-gradient(145deg, #15263d 0%, #101d30 74%);
    border-color: var(--line) !important;
    border-radius: 12px !important;
    box-shadow: 0 12px 32px rgba(0,0,0,.14);
}
[data-testid="stVerticalBlockBorderWrapper"] > div { padding: .65rem .9rem; }
.result-label {
    display: inline-block;
    margin: 0 0 .4rem;
    padding: .24rem .55rem;
    border-radius: 999px;
    font-size: .68rem;
    font-weight: 700;
    letter-spacing: .075em;
    line-height: 1.25;
    text-transform: uppercase;
}
.result-label.teal { background: var(--accent-soft); color: var(--accent); }
.result-label.blue { background: var(--blue-soft); color: var(--blue-ink); }
.result-label.amber { background: var(--amber-soft); color: var(--amber-ink); }
.result-label.violet { background: var(--violet-soft); color: var(--violet-ink); }
.result-label.rose { background: var(--rose-soft); color: var(--rose-ink); }
[data-testid="stMarkdownContainer"] { font-size: 1.04rem; }
[data-testid="stMarkdownContainer"] p { line-height: 1.5 !important; margin: .25rem 0 .42rem !important; }
[data-testid="stMarkdownContainer"] h1 { font-size: 1.7rem !important; margin: .25rem 0 .6rem !important; }
[data-testid="stMarkdownContainer"] h2 { font-size: 1.45rem !important; margin: .7rem 0 .35rem !important; }
[data-testid="stMarkdownContainer"] h3 { font-size: 1.24rem !important; margin: .65rem 0 .3rem !important; }
[data-testid="stMarkdownContainer"] h4 { font-size: 1.1rem !important; margin: .55rem 0 .25rem !important; }
[data-testid="stMarkdownContainer"] h5 { font-size: .88rem !important; margin: .55rem 0 .25rem !important; }
[data-testid="stMarkdownContainer"] ul, [data-testid="stMarkdownContainer"] ol { margin: .22rem 0 .48rem !important; padding-left: 1.35rem !important; }
[data-testid="stMarkdownContainer"] li { line-height: 1.5 !important; margin: 0 0 .28rem !important; padding-left: .1rem; }
[data-testid="stMarkdownContainer"] li > p { margin: .1rem 0 .25rem !important; }
[data-testid="stMarkdownContainer"] table { display: block; max-width: 100%; overflow-x: auto; }
[data-testid="stMarkdownContainer"] pre, [data-testid="stMarkdownContainer"] code { overflow-wrap: anywhere; white-space: pre-wrap; }
.stMarkdown strong { color: #f4fffd; font-weight: 700; }
.transcript-box {
    color: var(--ink);
    font-size: 1rem;
    line-height: 1.55;
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
    border-bottom: 1px solid rgba(154,173,196,.13);
    color: #c3d0df;
    font-size: .86rem;
}
.pipeline-row:last-child { border-bottom: 0; }
.pipeline-dot {
    width: .55rem;
    height: .55rem;
    border-radius: 50%;
    flex: 0 0 .55rem;
    background: #566b84;
}
.pipeline-row.active { color: var(--ink); font-weight: 700; }
.pipeline-row.active .pipeline-dot { background: var(--warning); box-shadow: 0 0 0 3px rgba(255,208,121,.18); }
.pipeline-row.done .pipeline-dot { background: var(--success); }
.pipeline-row.error { color: var(--danger); font-weight: 700; }
.pipeline-row.error .pipeline-dot { background: var(--danger); }
.pipeline-state { margin-left: auto; color: var(--muted); font-size: .72rem; font-weight: 500; }
.pipeline-row.active .pipeline-state { color: var(--warning); }
.pipeline-row.done .pipeline-state { color: var(--success); }
.pipeline-row.error .pipeline-state { color: var(--danger); }

.empty-state {
    background: linear-gradient(145deg, #15263d 0%, #101d30 74%);
    border: 1px solid var(--line);
    border-radius: 14px;
    padding: clamp(2rem, 5vw, 4.5rem) 2rem;
    text-align: center;
    box-shadow: 0 16px 38px rgba(0,0,0,.16);
}
.empty-mark {
    width: 3rem;
    height: 3rem;
    margin: 0 auto 1.1rem;
    border: 1px solid #28675f;
    border-radius: 12px;
    background: var(--accent-soft);
    color: var(--accent);
    display: grid;
    place-items: center;
    font-family: 'Manrope', sans-serif;
    font-size: 1.1rem;
    font-weight: 800;
}
.empty-state p {
    display: block;
    width: min(100%, 45rem);
    color: var(--muted);
    margin: .35rem auto 0 !important;
    text-align: center !important;
}

.chat-container {
    display: flex;
    flex-direction: column;
    gap: .75rem;
    margin: .5rem 0 1rem;
    min-width: 0;
    max-height: 34rem;
    overflow-y: auto;
    padding-right: .25rem;
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
.user-bubble { align-self: flex-end; background: var(--accent-soft); border-color: #28675f; }
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

.stTextInput input::placeholder, textarea::placeholder { color: #71859e !important; }
[data-baseweb="popover"] [role="listbox"], [data-baseweb="menu"] { background: #14243a !important; }
[data-baseweb="popover"] [role="option"] { color: var(--ink) !important; }
[data-baseweb="popover"] [role="option"]:hover { background: #1b3450 !important; }

@media (max-width: 850px) {
[data-testid="stMainBlockContainer"] { padding: 1.35rem 1.1rem 3rem; }
    .chat-bubble { max-width: 100%; }
    section[data-testid="stSidebar"] { width: min(88vw, 23rem) !important; min-width: min(88vw, 23rem) !important; }
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
    "chat_error": None,
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


def format_output_markdown(content: str) -> str:
    text = str(content).strip()
    lines = text.splitlines()
    if lines:
        title_match = re.fullmatch(r"\s*\*\*(.+?)\*\*\s*", lines[0])
        if title_match:
            lines[0] = f"### {title_match.group(1).strip()}"
    lines = [re.sub(r"^\s*[•●]\s+", "- ", line) for line in lines]
    return "\n".join(lines)


def render_content_card(title: str, content: str):
    label_colors = {
        "Meeting title": "blue",
        "Summary": "teal",
        "Action items": "amber",
        "Key decisions": "violet",
        "Open questions": "rose",
    }
    color = label_colors.get(title, "teal")
    with st.container(border=True):
        st.markdown(
            f'<div class="result-label {color}">{html.escape(title)}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(format_output_markdown(content))

                                                                                  
with st.sidebar:
    logo_data = base64.b64encode(
        (Path(__file__).parent / "assets" / "clipmind-mark.svg").read_bytes()
    ).decode("ascii")
    st.markdown(
        f'<div class="sidebar-brand"><img class="brand-logo" src="data:image/svg+xml;base64,{logo_data}" alt="">ClipMind</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<p class="sidebar-note">Turn a recording into clear notes and decisions.</p>', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Recording</div>', unsafe_allow_html=True)
    source = st.text_input(
        "Video or audio source",
        placeholder="Paste a YouTube link",
        label_visibility="collapsed",
    )
    uploaded_recording = st.file_uploader(
        "Or upload a recording",
        type=["mp3", "mp4", "wav", "m4a", "mpeg", "mpga", "webm", "mov", "avi", "mkv"],
    )
    st.markdown(
        '<div class="youtube-note"><strong>YouTube availability</strong>'
        'YouTube may block downloads from hosted servers. If a link cannot be '
        'processed, upload the audio or video file instead.</div>',
        unsafe_allow_html=True,
    )
    language = st.selectbox("Transcript language", ["english", "hinglish"], index=0)
    run_btn = st.button("Analyse", type="primary", use_container_width=True)
    st.markdown('<p class="sidebar-note">Paste a YouTube link or upload an audio/video file.</p>', unsafe_allow_html=True)
    pipeline_status_placeholder = st.empty()

render_pipeline_status(pipeline_status_placeholder, st.session_state.pipeline_steps)

st.markdown(
    f'<div class="main-brand"><img src="data:image/svg+xml;base64,{logo_data}" alt="">'
    '<div><h1>ClipMind</h1></div></div>',
    unsafe_allow_html=True,
)
st.markdown('<div class="page-subtitle">Clear notes and useful answers from every recording.</div>', unsafe_allow_html=True)
st.markdown("---")

                                                                                  
if run_btn:
    if source.strip() and uploaded_recording is not None:
        st.error("Choose a YouTube link or upload a recording, not both.")
    elif not source.strip() and uploaded_recording is None:
        st.error("Enter a YouTube link or upload a recording.")
    else:
        st.session_state.pipeline_done = False
        st.session_state.processing = True
        st.session_state.result = None
        st.session_state.chat_history = []
        st.session_state.chat_error = None
        st.session_state.pipeline_steps = {key: "pending" for key, _ in PIPELINE_STEPS}
        st.session_state.active_pipeline_step = None

        progress_placeholder = st.empty()
        uploaded_path = None
        chunks = []

        def update_step(key, state):
            st.session_state.pipeline_steps[key] = state
            st.session_state.active_pipeline_step = key if state == "active" else None
            render_pipeline_status(pipeline_status_placeholder, st.session_state.pipeline_steps)

        try:
            progress_placeholder.info("Analysis is running. You can follow each stage in the sidebar.")

            update_step("audio", "active")
            source_to_process = source.strip()
            if uploaded_recording is not None:
                suffix = Path(uploaded_recording.name).suffix or ".media"
                with tempfile.NamedTemporaryFile(prefix="clipmind-", suffix=suffix, delete=False) as media_file:
                    media_file.write(uploaded_recording.getbuffer())
                    uploaded_path = media_file.name
                source_to_process = uploaded_path
            chunks = process_input(source_to_process)
            update_step("audio", "done")

            update_step("transcript", "active")
            transcript = transcribe_all(chunks, language)
            if not transcript or not transcript.strip():
                raise ValueError(
                    "No speech was detected in the recording. Check that it has "
                    "clear audible speech, then try another recording."
                )
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
        finally:
            cleanup_input_chunks(chunks)
            if uploaded_path:
                uploaded_media = Path(uploaded_path)
                uploaded_media.unlink(missing_ok=True)

                                                                                   
if st.session_state.result:
    r = st.session_state.result

    content_col, question_col = st.columns([1.75, 1], gap="large")

    with content_col:
        render_content_card("Meeting title", r["title"])
        render_content_card("Summary", r["summary"])

        with st.expander("View full transcript", expanded=False):
            st.markdown(
                f'<div class="transcript-box">{html.escape(str(r["transcript"]))}</div>',
                unsafe_allow_html=True,
            )

        detail_col1, detail_col2 = st.columns(2, gap="medium")
        with detail_col1:
            render_content_card("Action items", r["action_items"])
        with detail_col2:
            render_content_card("Key decisions", r["key_decisions"])
        render_content_card("Open questions", r["open_questions"])

    with question_col:
        with st.container(border=True):
            st.markdown("### Ask ClipMind")
            st.markdown('<p class="sidebar-note">Answers are based on this meeting transcript.</p>', unsafe_allow_html=True)

            if st.session_state.chat_error:
                st.error(st.session_state.chat_error)
                st.session_state.chat_error = None

            if st.session_state.chat_history:
                chat_html = '<div class="chat-container">'
                for msg in st.session_state.chat_history:
                    if msg["role"] == "user":
                        label = "You"
                        bubble_class = "user-bubble"
                    else:
                        label = "ClipMind"
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
                    '<div class="sidebar-note">Ask about a decision, owner, deadline, or topic in the transcript.</div>',
                    unsafe_allow_html=True,
                )

            with st.form("meeting_question_form", clear_on_submit=True):
                user_input = st.text_input(
                    "Your question",
                    placeholder="Ask a question about the meeting",
                    label_visibility="collapsed",
                )
                send_btn = st.form_submit_button("Send", use_container_width=True)

            if st.session_state.chat_history and st.button("Clear conversation", type="secondary"):
                st.session_state.chat_history = []
                st.session_state.chat_error = None
                st.rerun()

    if send_btn and user_input.strip():
        st.session_state.chat_history.append({"role": "user", "content": user_input.strip()})
        try:
            with st.spinner("Thinking…"):
                answer = ask_question(r["rag_chain"], user_input.strip())
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
            st.session_state.chat_error = None
        except Exception as error:
            st.session_state.chat_error = display_error(error)
        st.rerun()

else:
    st.markdown("""
    <div class="empty-state">
        <div class="empty-mark">MI</div>
        <div class="eyebrow">MEETING NOTES, MADE CLEAR</div>
        <h2>Start with a recording</h2>
        <p>Add a YouTube link or local media file in the sidebar. ClipMind will create a transcript, summary, action items, and a searchable meeting chat.</p>
    </div>""", unsafe_allow_html=True)
