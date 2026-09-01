"""Streamlit interface.

This file deliberately contains no retrieval logic. It uploads files,
holds conversation state, and renders results -- everything else is a
call into rag/. Swapping this interface for a Telegram bot should not
require touching a single line inside the rag package.
"""

import streamlit as st

from rag import config
from rag.answer import answer_question
from rag.chunker import chunk_pages
from rag.embedder import Embedder
from rag.loader import EmptyPdfError, load_pdf
from rag.store import delete_source, get_client, get_collection, ingest, list_sources

st.set_page_config(page_title="Ask your PDF", page_icon="📄", layout="centered")


@st.cache_resource(show_spinner="Loading the embedding model...")
def get_embedder() -> Embedder:
    """Load the model once per session instead of once per keystroke.

    cache_resource is the right decorator here: it caches an object that
    should be shared and never copied. cache_data would try to serialise
    the model weights, which is both slow and wrong.
    """
    return Embedder()


@st.cache_resource
def get_store():
    """Open the vector store once and reuse the handle."""
    return get_collection(get_client())


def index_uploaded_file(uploaded_file, collection, embedder) -> None:
    """Save an uploaded PDF to data/ and add it to the vector store."""
    destination = config.DATA_DIR / uploaded_file.name
    destination.write_bytes(uploaded_file.getbuffer())

    try:
        pages = load_pdf(destination)
    except EmptyPdfError as error:
        st.error(str(error))
        return

    chunks = chunk_pages(pages, source=uploaded_file.name)

    with st.spinner(f"Indexing {len(chunks)} chunks..."):
        result = ingest(collection, chunks, embedder)

    if result["added"]:
        st.success(
            f"Indexed {uploaded_file.name}: "
            f"{result['added']} new chunks from {len(pages)} pages."
        )
    else:
        st.info(f"{uploaded_file.name} was already indexed. Nothing changed.")


# --- Sidebar: document management ---------------------------------------

collection = get_store()
embedder = get_embedder()

with st.sidebar:
    st.header("Documents")

    uploaded = st.file_uploader("Add a PDF", type="pdf")
    if uploaded is not None:
        if st.button("Index this file", type="primary", width="stretch"):
            index_uploaded_file(uploaded, collection, embedder)
            st.rerun()

    sources = list_sources(collection)

    if sources:
        st.divider()
        selected = st.selectbox(
            "Answer questions from",
            options=["All documents"] + sources,
        )
        active_source = None if selected == "All documents" else selected

        if active_source and st.button(f"Remove {active_source}"):
            removed = delete_source(collection, active_source)
            st.success(f"Removed {removed} chunks.")
            st.rerun()
    else:
        active_source = None

    st.divider()
    st.caption(
        f"Embeddings: {config.EMBEDDING_MODEL.split('/')[-1]} (local) · "
        f"Answers: {config.LLM_MODEL}"
    )

# --- Main pane: conversation --------------------------------------------

st.title("Ask your PDF")

if not sources:
    st.info(
        "Upload a PDF in the sidebar to get started. "
        "Answers will come only from the document, with a page reference "
        "under each one."
    )
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("hits"):
            with st.expander("Sources used"):
                for hit in message["hits"]:
                    st.markdown(
                        f"**Page {hit['label']}** · distance {hit['distance']:.3f}"
                    )
                    st.caption(hit["text"][:400] + "...")

question = st.chat_input("Ask something about the document")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the document..."):
            try:
                result = answer_question(
                    collection, embedder, question, source=active_source
                )
            except RuntimeError as error:
                st.error(str(error))
                st.stop()

        st.markdown(result["answer"])

        if result["hits"]:
            with st.expander("Sources used"):
                for hit in result["hits"]:
                    st.markdown(
                        f"**Page {hit['label']}** · distance {hit['distance']:.3f}"
                    )
                    st.caption(hit["text"][:400] + "...")

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result["answer"],
            "hits": result["hits"],
        }
    )
