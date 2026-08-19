from __future__ import annotations

import json
import os
from typing import Any

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")
QUERY_URL = f"{API_BASE_URL}/api/query"
STREAM_QUERY_URL = f"{API_BASE_URL}/api/query/stream"

st.set_page_config(page_title="InsuranceGuide", page_icon="IG", layout="wide")

st.markdown(
    """
    <style>
    :root {
        --ink: #07131d;
        --ink-soft: #0d202b;
        --glass: rgba(17, 42, 54, 0.72);
        --glass-light: rgba(30, 75, 88, 0.46);
        --line: rgba(151, 219, 211, 0.2);
        --text: #e7f3f0;
        --muted: #8eafb0;
        --teal: #7de0d2;
        --coral: #ff927b;
        --gold: #f1c777;
    }
    .stApp {
        background: radial-gradient(circle at 85% 8%, #173c49 0, transparent 28%),
                    radial-gradient(circle at 8% 78%, #122e3b 0, transparent 25%), var(--ink);
        color: var(--text);
    }
    .block-container { max-width: 1220px; padding: 2.8rem 3rem 4rem; }
    [data-testid="stSidebar"] { background: rgba(5, 18, 27, 0.93); border-right: 1px solid var(--line); }
    [data-testid="stSidebar"] > div:first-child { padding: 2rem 1.25rem; }
    h1, h2, h3, label, p { color: var(--text) !important; }
    .eyebrow { color: var(--teal); font-size: .72rem; font-weight: 700; letter-spacing: .16em; text-transform: uppercase; }
    .hero { padding: 1.2rem 0 2.4rem; }
    .hero h1 { font-size: clamp(2.4rem, 5vw, 4.8rem); line-height: .98; letter-spacing: -.04em; margin: .45rem 0 .9rem; }
    .hero-copy { max-width: 650px; color: var(--muted) !important; font-size: 1.05rem; }
    .status-chip { display: inline-block; border: 1px solid rgba(125, 224, 210, .35); border-radius: 999px; color: var(--teal); background: rgba(125, 224, 210, .08); padding: .35rem .7rem; font-size: .75rem; }
    .section-kicker { color: var(--gold) !important; font-size: .72rem; font-weight: 700; letter-spacing: .14em; text-transform: uppercase; margin-bottom: .35rem; }
    .answer { min-height: 170px; border: 1px solid var(--line); border-left: 4px solid var(--coral); border-radius: 14px; padding: 1.25rem 1.4rem; background: linear-gradient(130deg, rgba(20, 55, 67, .9), rgba(8, 26, 37, .82)); box-shadow: 0 18px 45px rgba(0,0,0,.2); }
    [data-testid="stTextArea"] textarea, [data-testid="stTextInput"] input { background: rgba(8, 27, 37, .82) !important; color: var(--text) !important; border: 1px solid var(--line) !important; border-radius: 12px !important; }
    .stButton > button { border: 1px solid rgba(125, 224, 210, .38); border-radius: 10px; background: rgba(30, 87, 91, .72); color: var(--text); font-weight: 700; }
    .stButton > button:hover { border-color: var(--teal); color: white; background: rgba(44, 112, 112, .85); }
    [data-testid="stExpander"] { border: 1px solid var(--line); border-radius: 11px; background: rgba(13, 36, 47, .62); }
    [data-testid="stFileUploader"] { border: 1px dashed rgba(125, 224, 210, .35); border-radius: 12px; padding: .4rem; background: rgba(17, 48, 58, .38); }
    code { color: var(--teal) !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">Policy intelligence / grounded answers</div>
      <h1>Understand the fine print<br>without the fog.</h1>
      <p class="hero-copy">Upload a policy, ask a precise question, and watch the answer arrive with the exact evidence behind it.</p>
      <span class="status-chip">● live retrieval enabled</span>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown('<div class="eyebrow">Workspace</div>', unsafe_allow_html=True)
    st.subheader("InsuranceGuide")
    st.caption("Your policy stays searchable for this session.")
    st.markdown('<div class="section-kicker">Backend connection</div>', unsafe_allow_html=True)
    st.code(API_BASE_URL, language="text")
    top_k = st.slider("Sources to retrieve", min_value=1, max_value=10, value=3)
    st.markdown('<div class="section-kicker">Policy document</div>', unsafe_allow_html=True)
    policy_file = st.file_uploader("Upload a PDF policy", type=["pdf"])
    upload_policy = st.button("Process policy", use_container_width=True)

    if upload_policy:
        if policy_file is None:
            st.warning("Choose a PDF policy first.")
        else:
            try:
                upload_response = requests.post(
                    f"{API_BASE_URL}/api/upload",
                    files={"file": (policy_file.name, policy_file.getvalue(), "application/pdf")},
                    timeout=180,
                )
                upload_response.raise_for_status()
                upload_result = upload_response.json()
                st.success(
                    f"Processed {upload_result.get('file', policy_file.name)}: "
                    f"{upload_result.get('chunks_indexed', 0)} searchable chunks."
                )
            except requests.RequestException as exc:
                detail = ""
                if getattr(exc, "response", None) is not None:
                    try:
                        detail = exc.response.json().get("detail", "")
                    except ValueError:
                        detail = exc.response.text
                st.error(f"Policy processing failed: {detail or exc}")

st.markdown('<div class="section-kicker">Ask your policy</div>', unsafe_allow_html=True)
question = st.text_area(
    "Question for your policy",
    placeholder="What does property insurance cover?",
    height=110,
)
submit = st.button("Ask the policy", type="primary", use_container_width=True)


def source_label(source: dict[str, Any], index: int) -> str:
    marker = source.get("marker", f"[{index}]")
    name = source.get("source") or "Unknown source"
    chunk = source.get("chunk_index", "?")
    score = source.get("score")
    score_text = f" | relevance {score}" if score is not None else ""
    return f"{marker} {name} | chunk {chunk}{score_text}"


def render_sources(sources: list[dict[str, Any]]) -> None:
    st.subheader("Grounding sources")
    if not sources:
        st.info("No source chunks were retrieved for this question.")
        return

    for index, source in enumerate(sources, start=1):
        with st.expander(source_label(source, index), expanded=index == 1):
            metadata = {
                key: value
                for key, value in source.items()
                if key not in {"source", "chunk_index", "score", "url", "link"}
                and value is not None
            }
            if source.get("url") or source.get("link"):
                st.markdown(f"[Open source]({source.get('url') or source.get('link')})")
            if source.get("text"):
                st.markdown("**Cited content**")
                st.code(source["text"], language="text")
            if metadata:
                st.json(metadata)


if submit:
    question = question.strip()
    if not question:
        st.error("Enter a question before submitting.")
    else:
        left, right = st.columns([1.6, 1], gap="large")
        with left:
            st.markdown('<div class="section-kicker">Response / live stream</div>', unsafe_allow_html=True)
            st.subheader("Grounded answer")
            answer_placeholder = st.empty()
            answer_placeholder.info("Retrieving sources and starting the answer...")
        with right:
            sources_placeholder = st.empty()

        answer = ""
        sources: list[dict[str, Any]] = []
        stream_failed = False
        try:
            with requests.post(
                STREAM_QUERY_URL,
                json={"question": question, "k": top_k},
                stream=True,
                timeout=(10, 120),
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines(decode_unicode=True):
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                    except ValueError:
                        stream_failed = True
                        st.error("The RAG API returned an invalid streaming event.")
                        break

                    if event.get("type") == "sources":
                        sources = event.get("sources", [])
                        with sources_placeholder.container():
                            render_sources(sources)
                    elif event.get("type") == "token":
                        answer += event.get("text", "")
                        answer_placeholder.markdown(answer)
                    elif event.get("type") == "error":
                        stream_failed = True
                        st.error(event.get("detail", "The answer stream was interrupted."))
                        break
        except requests.RequestException as exc:
            stream_failed = True
            st.error(f"Could not reach the RAG API: {exc}")

        if stream_failed and answer:
            st.warning("The stream stopped early. The partial answer remains visible above.")
        elif not stream_failed and not answer:
            answer_placeholder.info("The API returned no answer.")
