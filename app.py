import os
import io
import hashlib
import numpy as np
import streamlit as st
import faiss

from pypdf import PdfReader
from groq import Groq
from sentence_transformers import SentenceTransformer


# -----------------------------
# App configuration
# -----------------------------
st.set_page_config(
    page_title="RAG PDF Assistant",
    page_icon="📚",
    layout="wide",
)

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL = "openai/gpt-oss-120b"
CHUNK_SIZE = 450
CHUNK_OVERLAP = 80
TOP_K_DEFAULT = 5


# -----------------------------
# Custom UI
# -----------------------------
st.markdown(
    """
    <style>
        .block-container {
            max-width: 1150px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }
        .hero {
            padding: 1.4rem 1.6rem;
            border: 1px solid rgba(128,128,128,.22);
            border-radius: 18px;
            margin-bottom: 1rem;
        }
        .hero h1 {
            margin: 0 0 .35rem 0;
            font-size: 2.2rem;
        }
        .hero p {
            margin: 0;
            opacity: .78;
            font-size: 1rem;
        }
        .metric-box {
            border: 1px solid rgba(128,128,128,.18);
            border-radius: 14px;
            padding: .9rem 1rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <h1>📚 RAG PDF Assistant</h1>
        <p>Upload a PDF, build a FAISS vector index, and ask grounded questions using Groq.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# Helpers
# -----------------------------
@st.cache_resource(show_spinner="Loading open-source embedding model...")
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)


def get_groq_key():
    """Read GROQ_API_KEY from Streamlit secrets or environment variables."""
    try:
        if "GROQ_API_KEY" in st.secrets:
            return st.secrets["GROQ_API_KEY"]
    except Exception:
        pass

    return os.environ.get("GROQ_API_KEY")


def extract_pdf_pages(pdf_bytes: bytes):
    """Extract text page-by-page from a PDF."""
    reader = PdfReader(io.BytesIO(pdf_bytes))
    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = " ".join(text.split())

        if text.strip():
            pages.append(
                {
                    "page": page_number,
                    "text": text.strip(),
                }
            )

    return pages


def token_chunk_pages(pages, tokenizer, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """
    Split PDF text into token-aware chunks.

    Tokenization uses the tokenizer that belongs to the open-source
    SentenceTransformer embedding model.
    """
    chunks = []

    for page_data in pages:
        page_number = page_data["page"]
        text = page_data["text"]

        token_ids = tokenizer.encode(
            text,
            add_special_tokens=False,
            truncation=False,
        )

        if not token_ids:
            continue

        start = 0
        chunk_number = 1

        while start < len(token_ids):
            end = min(start + chunk_size, len(token_ids))
            current_ids = token_ids[start:end]

            chunk_text = tokenizer.decode(
                current_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            ).strip()

            if chunk_text:
                chunks.append(
                    {
                        "page": page_number,
                        "chunk": chunk_number,
                        "text": chunk_text,
                        "token_count": len(current_ids),
                    }
                )

            if end >= len(token_ids):
                break

            start = max(end - overlap, start + 1)
            chunk_number += 1

    return chunks


def build_faiss_index(chunks, embedding_model):
    """Create normalized embeddings and store them in an in-memory FAISS index."""
    texts = [item["text"] for item in chunks]

    embeddings = embedding_model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=True,
    ).astype("float32")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    return index


def retrieve(query, index, chunks, embedding_model, top_k=5):
    """Retrieve the most similar chunks using cosine similarity."""
    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=True,
    ).astype("float32")

    k = min(top_k, len(chunks))
    scores, indices = index.search(query_embedding, k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue

        item = chunks[int(idx)].copy()
        item["score"] = float(score)
        results.append(item)

    return results


def build_context(retrieved_chunks):
    """Format retrieved chunks for the LLM."""
    context_parts = []

    for i, item in enumerate(retrieved_chunks, start=1):
        context_parts.append(
            f"[Source {i} | Page {item['page']} | Chunk {item['chunk']}]\n"
            f"{item['text']}"
        )

    return "\n\n".join(context_parts)


def ask_groq(api_key, question, context):
    """Generate a grounded answer using Groq."""
    client = Groq(api_key=api_key)

    system_prompt = (
        "You are a document question-answering assistant. "
        "Answer only from the supplied PDF context. "
        "If the answer is not supported by the context, clearly say that "
        "the information was not found in the uploaded document. "
        "Do not invent facts. "
        "When useful, mention page numbers using the source labels supplied in the context. "
        "Keep the answer clear and well structured."
    )

    user_prompt = f"""PDF CONTEXT:
{context}

QUESTION:
{question}

Answer the question using only the PDF context above."""

    completion = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=1200,
    )

    return completion.choices[0].message.content


def reset_document_state():
    keys = [
        "document_hash",
        "document_name",
        "pages",
        "chunks",
        "faiss_index",
        "messages",
    ]
    for key in keys:
        st.session_state.pop(key, None)


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("⚙️ Settings")

    saved_key = get_groq_key()
    api_key_input = st.text_input(
        "Groq API Key",
        type="password",
        placeholder="gsk_...",
        help=(
            "For Streamlit Cloud, storing GROQ_API_KEY in Secrets is safer. "
            "If a secret is configured, you can leave this field empty."
        ),
    )

    api_key = api_key_input.strip() or saved_key

    top_k = st.slider(
        "Retrieved chunks",
        min_value=2,
        max_value=10,
        value=TOP_K_DEFAULT,
        help="How many semantically similar PDF chunks are sent to the LLM.",
    )

    st.divider()
    st.caption(f"LLM: `{GROQ_MODEL}`")
    st.caption(f"Embeddings: `{EMBEDDING_MODEL}`")
    st.caption("Vector DB: `FAISS`")


# -----------------------------
# PDF upload and indexing
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload a PDF document",
    type=["pdf"],
    help="Text-based PDFs work best. Scanned/image-only PDFs require OCR, which is not included in this 2-file version.",
)

if uploaded_file is None:
    st.info("👆 Upload a PDF to build the RAG knowledge base.")
    st.stop()


pdf_bytes = uploaded_file.getvalue()
current_hash = hashlib.sha256(pdf_bytes).hexdigest()

if st.session_state.get("document_hash") != current_hash:
    reset_document_state()

    try:
        embedding_model = load_embedding_model()

        with st.status("Processing PDF...", expanded=True) as status:
            st.write("1. Extracting text from PDF...")
            pages = extract_pdf_pages(pdf_bytes)

            if not pages:
                status.update(label="No readable text found", state="error")
                st.error(
                    "I could not extract readable text from this PDF. "
                    "It may be scanned/image-only. Try a text-based PDF or add OCR later."
                )
                st.stop()

            st.write("2. Tokenizing and creating overlapping chunks...")
            chunks = token_chunk_pages(
                pages,
                embedding_model.tokenizer,
                CHUNK_SIZE,
                CHUNK_OVERLAP,
            )

            if not chunks:
                status.update(label="No chunks created", state="error")
                st.error("No usable text chunks could be created from the document.")
                st.stop()

            st.write("3. Creating open-source embeddings...")
            st.write("4. Building FAISS vector index...")
            index = build_faiss_index(chunks, embedding_model)

            status.update(
                label="PDF indexed successfully",
                state="complete",
                expanded=False,
            )

        st.session_state.document_hash = current_hash
        st.session_state.document_name = uploaded_file.name
        st.session_state.pages = pages
        st.session_state.chunks = chunks
        st.session_state.faiss_index = index
        st.session_state.messages = []

    except Exception as exc:
        st.error(f"PDF processing failed: {exc}")
        st.stop()


# -----------------------------
# Knowledge-base summary
# -----------------------------
pages = st.session_state.pages
chunks = st.session_state.chunks

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("📄 Pages with text", len(pages))
with col2:
    st.metric("🧩 Chunks", len(chunks))
with col3:
    st.metric(
        "🔤 Approx. indexed tokens",
        f"{sum(c['token_count'] for c in chunks):,}",
    )

st.success(f"Knowledge base ready: **{st.session_state.document_name}**")


# -----------------------------
# Chat
# -----------------------------
st.subheader("💬 Ask your PDF")

for message in st.session_state.get("messages", []):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Ask something about the uploaded PDF...")

if question:
    if not api_key:
        st.error(
            "Please enter your Groq API key in the sidebar, or configure "
            "`GROQ_API_KEY` in Streamlit Secrets."
        )
        st.stop()

    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user"):
        st.markdown(question)

    try:
        embedding_model = load_embedding_model()

        with st.chat_message("assistant"):
            with st.spinner("Retrieving relevant context and generating answer..."):
                retrieved = retrieve(
                    question,
                    st.session_state.faiss_index,
                    st.session_state.chunks,
                    embedding_model,
                    top_k=top_k,
                )

                context = build_context(retrieved)
                answer = ask_groq(api_key, question, context)

            st.markdown(answer)

            with st.expander("🔎 Retrieved source chunks"):
                for i, item in enumerate(retrieved, start=1):
                    st.markdown(
                        f"**Source {i} · Page {item['page']} · "
                        f"Chunk {item['chunk']} · Similarity {item['score']:.3f}**"
                    )
                    st.write(item["text"])
                    st.divider()

        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )

    except Exception as exc:
        st.error(f"Groq request failed: {exc}")
