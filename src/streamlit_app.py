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

st.set_page_config(page_title="InsuranceGuide", page_icon="🛡️", layout="wide")

st.markdown(
    """
    <style>
    .stApp { background: #f5f1e8; }
    .block-container { max-width: 1100px; padding-top: 3rem; }
    h1, h2, h3 { color: #173f3f; }
    .answer { border-left: 4px solid #e07a5f; padding: 1rem 1.25rem; background: #fffdf8; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("InsuranceGuide")
st.caption("Ask a policy question and inspect the documents that ground the answer.")

with st.sidebar:
    st.subheader("Connection")
    st.code(API_BASE_URL, language="text")
    top_k = st.slider("Sources to retrieve", min_value=1, max_value=10, value=3)

question = st.text_area(
    "Question",
    placeholder="What does property insurance cover?",
    height=110,
)
submit = st.button("Ask InsuranceGuide", type="primary", use_container_width=True)


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
