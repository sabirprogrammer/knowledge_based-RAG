import os
import io
import hashlib
import streamlit as st
import faiss

from pypdf import PdfReader
from groq import Groq
from sentence_transformers import SentenceTransformer


# -----------------------------
# App configuration
# -----------------------------
st.set_page_config(
    page_title="DocMind AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL = "openai/gpt-oss-120b"
CHUNK_SIZE = 450
CHUNK_OVERLAP = 80
TOP_K_DEFAULT = 5


# -----------------------------
# Premium UI
# -----------------------------
st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] {
        background:
            radial-gradient(circle at 10% 10%, rgba(99,102,241,.10), transparent 30%),
            radial-gradient(circle at 90% 15%, rgba(16,185,129,.08), transparent 28%);
    }

    .block-container {
        max-width: 1180px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    [data-testid="stSidebar"] {
        border-right: 1px solid rgba(148,163,184,.14);
    }

    .rag-hero {
        position: relative;
        overflow: hidden;
        padding: 2rem 2.1rem;
        border-radius: 24px;
        border: 1px solid rgba(148,163,184,.18);
        background: linear-gradient(135deg, rgba(99,102,241,.13), rgba(16,185,129,.07));
        box-shadow: 0 18px 55px rgba(0,0,0,.10);
        margin-bottom: 1.4rem;
    }

    .rag-badge {
        display: inline-flex;
        align-items: center;
        gap: .45rem;
        padding: .38rem .7rem;
        border-radius: 999px;
        background: rgba(99,102,241,.12);
        border: 1px solid rgba(99,102,241,.22);
        font-size: .82rem;
        font-weight: 700;
        margin-bottom: .9rem;
    }

    .rag-hero h1 {
        margin: 0;
        font-size: clamp(2rem, 4vw, 3.25rem);
        line-height: 1.06;
        letter-spacing: -.04em;
    }

    .rag-hero p {
        margin: .85rem 0 0 0;
        max-width: 760px;
        opacity: .76;
        font-size: 1.05rem;
        line-height: 1.7;
    }

    .tech-row {
        display: flex;
        flex-wrap: wrap;
        gap: .55rem;
        margin-top: 1.15rem;
    }

    .tech-pill {
        padding: .42rem .72rem;
        border-radius: 999px;
        border: 1px solid rgba(148,163,184,.20);
        background: rgba(15,23,42,.08);
        font-size: .82rem;
        font-weight: 650;
    }

    .side-brand {
        padding: .5rem 0 1rem 0;
    }

    .side-brand h2 {
        margin: 0;
        font-size: 1.35rem;
        letter-spacing: -.02em;
    }

    .side-brand p {
        margin: .35rem 0 0 0;
        opacity: .65;
        font-size: .88rem;
    }

    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1.1rem;
    }

    .side-card {
        padding: 1rem;
        margin: .65rem 0 1rem 0;
        border-radius: 18px;
        border: 1px solid rgba(148,163,184,.14);
        background: linear-gradient(145deg, rgba(99,102,241,.10), rgba(16,185,129,.05));
    }

    .side-card-title {
        font-weight: 800;
        font-size: .88rem;
        margin-bottom: .35rem;
    }

    .side-card-text {
        opacity: .66;
        font-size: .82rem;
        line-height: 1.55;
    }

    [data-testid="stSidebar"] [data-testid="stSlider"] {
        padding: .25rem .2rem .55rem .2rem;
    }

    [data-testid="stSidebar"] hr {
        margin: 1rem 0;
        opacity: .16;
    }

    div[data-testid="stMetric"] {
        border: 1px solid rgba(148,163,184,.17);
        border-radius: 18px;
        padding: 1rem 1.1rem;
        background: rgba(148,163,184,.04);
    }

    [data-testid="stFileUploader"] {
        border-radius: 20px;
    }

    [data-testid="stFileUploaderDropzone"] {
        border-radius: 18px;
        border: 1px dashed rgba(99,102,241,.45);
        padding: 1.2rem;
    }

    div[data-testid="stChatMessage"] {
        border: 1px solid rgba(148,163,184,.13);
        border-radius: 18px;
        padding: .35rem .5rem;
        margin-bottom: .65rem;
    }

    .empty-card {
        text-align: center;
        padding: 2.25rem 1rem;
        border: 1px dashed rgba(148,163,184,.24);
        border-radius: 20px;
        background: rgba(148,163,184,.035);
        margin-top: .8rem;
    }

    .empty-icon {
        font-size: 2.25rem;
        margin-bottom: .55rem;
    }

    .empty-card h3 {
        margin: .25rem 0;
    }

    .empty-card p {
        margin: .4rem auto 0;
        max-width: 560px;
        opacity: .65;
        line-height: 1.6;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# Helpers
# -----------------------------
@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)


def get_groq_key():
    """Read the API key only from Streamlit Secrets or environment variables."""
    try:
        if "GROQ_API_KEY" in st.secrets:
            return st.secrets["GROQ_API_KEY"]
    except Exception:
        pass

    return os.environ.get("GROQ_API_KEY")


def extract_pdf_pages(pdf_bytes: bytes):
    reader = PdfReader(io.BytesIO(pdf_bytes))
    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = " ".join(text.split())

        if text.strip():
            pages.append({"page": page_number, "text": text.strip()})

    return pages


def token_chunk_pages(pages, tokenizer, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []

    for page_data in pages:
        token_ids = tokenizer.encode(
            page_data["text"],
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
                        "page": page_data["page"],
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
    texts = [item["text"] for item in chunks]

    embeddings = embedding_model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=True,
    ).astype("float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index


def retrieve(query, index, chunks, embedding_model, top_k=5):
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
    return "\n\n".join(
        (
            f"[Source {i} | Page {item['page']} | Chunk {item['chunk']}]\n"
            f"{item['text']}"
        )
        for i, item in enumerate(retrieved_chunks, start=1)
    )


def ask_groq(api_key, question, context):
    client = Groq(api_key=api_key)

    system_prompt = (
        "You are a document question-answering assistant. "
        "Answer only from the supplied PDF context. "
        "If the answer is not supported by the context, clearly say that "
        "the information was not found in the uploaded document. "
        "Do not invent facts. Mention page numbers when useful. "
        "Keep the answer clear, concise, and well structured."
    )

    completion = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    f"PDF CONTEXT:\n{context}\n\n"
                    f"QUESTION:\n{question}\n\n"
                    "Answer using only the PDF context above."
                ),
            },
        ],
        temperature=0.2,
        max_tokens=1200,
    )

    return completion.choices[0].message.content


def reset_document_state():
    for key in [
        "document_hash",
        "document_name",
        "pages",
        "chunks",
        "faiss_index",
        "messages",
    ]:
        st.session_state.pop(key, None)


# -----------------------------
# Secrets / runtime config
# -----------------------------
api_key = get_groq_key()


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.markdown(
        """
        <div class="side-brand">
            <h2>🧠 DocMind AI</h2>
            <p>Private document intelligence</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="side-card">
            <div class="side-card-title">✨ Smart PDF Search</div>
            <div class="side-card-text">
                Upload a document, then ask questions naturally. The app retrieves
                the most relevant passages before generating each answer.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### 🎯 Retrieval depth")
    top_k = st.slider(
        "Relevant chunks",
        min_value=2,
        max_value=10,
        value=TOP_K_DEFAULT,
        help="Choose how many relevant passages should be used for each answer.",
        label_visibility="collapsed",
    )

    st.caption("Fewer chunks = focused answers · More chunks = broader context")

    st.markdown("---")
    st.markdown(
        """
        <div class="side-card">
            <div class="side-card-title">🔒 Private by design</div>
            <div class="side-card-text">
                API credentials are handled through Streamlit Secrets and are never
                requested from visitors inside the app.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------
# Main header
# -----------------------------
st.markdown(
    """
    <div class="rag-hero">
        <div class="rag-badge">✦ Retrieval-Augmented Generation</div>
        <h1>Chat with your PDF.<br>Grounded in your document.</h1>
        <p>
            Upload a PDF and DocMind AI will extract its text, create token-aware
            chunks, generate open-source embeddings, index them with FAISS, and use
            Groq to answer questions from the most relevant context.
        </p>
        <div class="tech-row">
            <span class="tech-pill">📄 PyPDF</span>
            <span class="tech-pill">🧩 Token Chunking</span>
            <span class="tech-pill">🧠 MiniLM Embeddings</span>
            <span class="tech-pill">⚡ FAISS</span>
            <span class="tech-pill">🚀 Groq GPT-OSS</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if not api_key:
    st.warning(
        "Groq API key is not configured. In Streamlit Cloud open "
        "**Manage app → Settings → Secrets** and add "
        '`GROQ_API_KEY = "gsk_..."`.'
    )


# -----------------------------
# Upload / indexing
# -----------------------------
st.markdown("### 📁 Add a knowledge source")
st.caption("Upload a text-based PDF. Your document is processed in the current app session.")

uploaded_file = st.file_uploader(
    "Drop your PDF here",
    type=["pdf"],
    label_visibility="collapsed",
)

if uploaded_file is None:
    st.markdown(
        """
        <div class="empty-card">
            <div class="empty-icon">📚</div>
            <h3>Your knowledge base is empty</h3>
            <p>
                Upload a PDF above. The app will extract the text, create embeddings,
                and build a searchable FAISS index automatically.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()


pdf_bytes = uploaded_file.getvalue()
current_hash = hashlib.sha256(pdf_bytes).hexdigest()

if st.session_state.get("document_hash") != current_hash:
    reset_document_state()

    try:
        embedding_model = load_embedding_model()

        with st.status("Building your knowledge base...", expanded=True) as status:
            st.write("📄 Extracting PDF text")
            pages = extract_pdf_pages(pdf_bytes)

            if not pages:
                status.update(label="No readable text found", state="error")
                st.error(
                    "No readable text was found. This PDF may be scanned or image-only."
                )
                st.stop()

            st.write("🧩 Creating token-aware chunks")
            chunks = token_chunk_pages(
                pages,
                embedding_model.tokenizer,
                CHUNK_SIZE,
                CHUNK_OVERLAP,
            )

            if not chunks:
                status.update(label="No chunks created", state="error")
                st.error("No usable chunks could be created from this document.")
                st.stop()

            st.write("🧠 Generating open-source embeddings")
            st.write("⚡ Building FAISS vector index")
            index = build_faiss_index(chunks, embedding_model)

            status.update(
                label="Knowledge base ready",
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
# Knowledge base overview
# -----------------------------
pages = st.session_state.pages
chunks = st.session_state.chunks

st.markdown("### ✨ Knowledge base ready")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Pages", len(pages))
m2.metric("Chunks", len(chunks))
m3.metric("Indexed tokens", f"{sum(c['token_count'] for c in chunks):,}")
m4.metric("Retriever", f"Top {top_k}")

st.caption(f"Currently indexed: **{st.session_state.document_name}**")


# -----------------------------
# Chat
# -----------------------------
st.markdown("---")
st.markdown("### 💬 Ask your document")
st.caption("Answers are generated from the most relevant chunks retrieved from your PDF.")

if not st.session_state.get("messages"):
    st.info("Try a question like: **Summarize the main ideas of this document.**")

for message in st.session_state.get("messages", []):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Ask a question about this PDF...")

if question:
    if not api_key:
        st.error(
            "Groq API key is missing. Add GROQ_API_KEY in Streamlit Cloud Secrets, "
            "then reboot the app."
        )
        st.stop()

    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    try:
        embedding_model = load_embedding_model()

        with st.chat_message("assistant"):
            with st.spinner("Searching the document and generating an answer..."):
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

            with st.expander("🔎 View retrieved evidence"):
                for i, item in enumerate(retrieved, start=1):
                    st.markdown(
                        f"**Source {i} · Page {item['page']} · "
                        f"Chunk {item['chunk']} · Similarity {item['score']:.3f}**"
                    )
                    st.write(item["text"])
                    if i != len(retrieved):
                        st.divider()

        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )

    except Exception as exc:
        st.error(f"Groq request failed: {exc}")
