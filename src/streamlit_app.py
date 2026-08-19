from __future__ import annotations

import os
from typing import Any

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")
QUERY_URL = f"{API_BASE_URL}/api/query"

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
    name = source.get("source") or "Unknown source"
    chunk = source.get("chunk_index", "?")
    score = source.get("score")
    score_text = f" | relevance {score}" if score is not None else ""
    return f"{index}. {name} | chunk {chunk}{score_text}"


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
            if metadata:
                st.json(metadata)


if submit:
    question = question.strip()
    if not question:
        st.error("Enter a question before submitting.")
    else:
        with st.spinner("Retrieving grounded answer..."):
            try:
                response = requests.post(
                    QUERY_URL,
                    json={"question": question, "k": top_k},
                    timeout=60,
                )
                response.raise_for_status()
                payload = response.json()
            except requests.RequestException as exc:
                st.error(f"Could not reach the RAG API: {exc}")
            except ValueError:
                st.error("The RAG API returned invalid JSON.")
            else:
                if payload.get("status") != "success" or not payload.get("answer"):
                    st.error(payload.get("detail", "The API did not return a grounded answer."))
                else:
                    answer, sources = payload["answer"], payload.get("sources", [])
                    left, right = st.columns([1.6, 1], gap="large")
                    with left:
                        st.subheader("Grounded answer")
                        st.markdown('<div class="answer">', unsafe_allow_html=True)
                        st.markdown(answer)
                        st.markdown('</div>', unsafe_allow_html=True)
                    with right:
                        render_sources(sources)
                    metadata = payload.get("metadata", {})
                    if metadata:
                        st.caption(
                            f"Retrieved {metadata.get('retrieved_chunks', len(sources))} source(s) "
                            f"with top-k={metadata.get('top_k', top_k)}."
                        )
