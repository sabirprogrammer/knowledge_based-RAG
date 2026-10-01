# 🧠 DocMind AI

> **A RAG-based AI document assistant that lets you chat with your PDFs using open-source embeddings, FAISS, Groq, and Streamlit.**

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://knowledgebased-assistantr.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![FAISS](https://img.shields.io/badge/Vector%20Store-FAISS-4B8BBE?style=for-the-badge)
![Groq](https://img.shields.io/badge/LLM-Groq-orange?style=for-the-badge)

## 🚀 Live Demo

**Try DocMind AI here:**  
https://knowledgebased-assistantr.streamlit.app/

---

## ✨ About DocMind AI

**DocMind AI** is a lightweight Retrieval-Augmented Generation (RAG) application for asking questions about PDF documents.

Instead of sending an entire document directly to a language model, DocMind AI:

1. Extracts text from the uploaded PDF
2. Tokenizes and divides the document into overlapping chunks
3. Generates semantic embeddings using an open-source embedding model
4. Stores those embeddings in a FAISS vector index
5. Retrieves the most relevant chunks for every question
6. Sends only the retrieved context to a Groq-hosted language model
7. Returns a grounded answer based on the uploaded document

This makes the application more focused, efficient, and useful for document-based question answering.

---

## 🌟 Features

- 📄 Upload and process PDF documents
- 🧩 Token-aware text chunking with overlap
- 🧠 Open-source Sentence Transformers embeddings
- ⚡ Fast semantic retrieval with FAISS
- 🤖 Groq-powered answer generation
- 💬 Chat-style Streamlit interface
- 🔎 View retrieved source chunks and similarity scores
- 📑 Page-aware document context
- 🔐 Groq API key stored securely with Streamlit Secrets
- ☁️ Ready for Streamlit Community Cloud deployment
- 🪶 Lightweight two-file architecture

---

## 🏗️ RAG Architecture

```text
User uploads PDF
        │
        ▼
   PDF Extraction
      (PyPDF)
        │
        ▼
 Tokenization + Chunking
        │
        ▼
 Open-source Embeddings
 all-MiniLM-L6-v2
        │
        ▼
   FAISS Vector Index
        │
        ▼
    User Question
        │
        ▼
  Question Embedding
        │
        ▼
 Similarity Retrieval
        │
        ▼
 Relevant PDF Chunks
        │
        ▼
 Groq / GPT-OSS Model
        │
        ▼
   Grounded Answer
```

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Frontend | Streamlit |
| Language | Python |
| PDF Extraction | PyPDF |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Vector Store | FAISS |
| LLM API | Groq |
| LLM | openai/gpt-oss-120b |
| Deployment | Streamlit Community Cloud |

---

## 📁 Project Structure

```text
knowledge_based-RAG/
├── app.py
├── requirements.txt
└── README.md
```

The project intentionally keeps the architecture small and easy to understand.

---

## ⚙️ How It Works

### 1. PDF extraction

The uploaded PDF is read using **PyPDF**. Text is extracted page by page so retrieved chunks can retain page information.

### 2. Token-aware chunking

Extracted text is tokenized using the tokenizer from the embedding model.

Current configuration:

```python
CHUNK_SIZE = 450
CHUNK_OVERLAP = 80
```

Overlapping chunks help preserve information that appears near chunk boundaries.

### 3. Embeddings

DocMind AI uses:

```text
sentence-transformers/all-MiniLM-L6-v2
```

The embedding model runs locally in the application and converts each text chunk into a semantic vector.

### 4. FAISS vector search

Normalized embeddings are stored in an in-memory **FAISS** index.

When a user asks a question, DocMind AI embeds the question and finds the most semantically relevant document chunks.

### 5. Groq answer generation

The retrieved document context and question are sent to:

```text
openai/gpt-oss-120b
```

through the Groq API.

The model is instructed to answer only from the provided document context.

---

## 💻 Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/sabirprogrammer/knowledge_based-RAG.git
cd knowledge_based-RAG
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure your Groq API key

For local testing, set the environment variable:

**Windows CMD**

```bash
set GROQ_API_KEY=your_groq_api_key
```

**Windows PowerShell**

```powershell
$env:GROQ_API_KEY="your_groq_api_key"
```

**Linux/macOS**

```bash
export GROQ_API_KEY="your_groq_api_key"
```

### 4. Start the app

```bash
streamlit run app.py
```

Then open the local Streamlit URL shown in your terminal.

---

## ☁️ Deploy on Streamlit Community Cloud

1. Fork or upload this project to GitHub.
2. Open **Streamlit Community Cloud**.
3. Create a new app.
4. Select this repository.
5. Set the main file to:

```text
app.py
```

6. Open **Advanced settings / Secrets**.
7. Add:

```toml
GROQ_API_KEY = "your_groq_api_key"
```

8. Deploy the application.

Never commit your real API key to GitHub.

---

## 🔐 Security

DocMind AI does not ask visitors to type the Groq API key into the application interface.

The deployed application reads the key from **Streamlit Secrets**, keeping credentials separate from the public GitHub code.

---

## ⚠️ Current Limitations

- Best suited for text-based PDFs
- Scanned/image-only PDFs require OCR support
- The FAISS index exists only for the current Streamlit session
- Very large documents may take longer to embed on first upload
- This version supports one uploaded PDF at a time

---

## 🗺️ Possible Future Improvements

- OCR for scanned PDFs
- Multiple PDF support
- Persistent vector database
- Conversation-aware retrieval
- Citation links to exact PDF pages
- Hybrid semantic + keyword search
- Document collections and folders
- User accounts and saved chats
- More embedding/model options

---

## 📌 Use Cases

DocMind AI can be useful for:

- 📚 Students studying lecture notes and textbooks
- 🔬 Researchers exploring papers
- 📑 Reading technical reports
- 🧾 Understanding policies and documentation
- 💼 Reviewing business documents
- 🧠 Building and learning practical RAG systems

---

## 👨‍💻 Author

Built by **Sanaullah Sabir**

GitHub: [@sabirprogrammer](https://github.com/sabirprogrammer)

---

## ⭐ Support

If you find DocMind AI useful, consider giving the repository a **star** ⭐.

**Live App:** https://knowledgebased-assistantr.streamlit.app/
