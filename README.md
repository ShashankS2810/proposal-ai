# ProposalAI — Review & Improvement System

> **Fully offline** proposal review powered by Ollama + RAG (LangChain + ChromaDB)

---

## Architecture

```
User Uploads File (.pdf / .docx / .txt / .csv)
              ↓
       Document Loader
    (PyPDFLoader / Docx2txt / TextLoader / CSVLoader)
              ↓
       Text Chunking
    (RecursiveCharacterTextSplitter — 512 tokens, 64 overlap)
              ↓
     Embeddings Generation
    (nomic-embed-text  OR  mxbai-embed-large  via Ollama)
              ↓
       ChromaDB Storage
    (persisted at ~/.proposal_rag/chroma_db)
              ↓
     User Action (Review / Improve / Q&A / Score)
              ↓
     Similarity Search (MMR — diverse top-k chunks)
              ↓
       Relevant Chunks as Context
              ↓
       Ollama LLM (llama3.2 / mistral / phi3 / …)
              ↓
     Streamed Response → PyQt6 UI
```

---

## Tech Stack

| Purpose           | Technology                        |
|-------------------|-----------------------------------|
| Local LLM         | Ollama                            |
| RAG Framework     | LangChain                         |
| Vector DB         | ChromaDB (persisted locally)      |
| Embeddings        | nomic-embed-text / mxbai-embed-large |
| File Reading      | PyPDFLoader, Docx2txtLoader, etc. |
| UI                | PyQt6                             |
| Data Processing   | Pandas                            |

---

## Quick Start

### 1. Prerequisites

- Python 3.11+
- [Ollama](https://ollama.ai) installed and running

### 2. Setup (once)

```bash
cd proposal_rag
python setup.py
```

This installs all Python packages and pulls the required Ollama models.

### 3. Run

```bash
python main.py
```

---

## Features

### 📋 Review Tab
- Full structured review with scores, strengths, weaknesses
- Actionable recommendations
- Streams tokens in real time

### ✨ Improve Tab
- Rewrites the proposal to be more compelling
- Optional "focus" field to target specific sections (e.g. "executive summary")
- Preserves author intent

### 💬 Ask Questions Tab
- RAG-powered Q&A grounded in the document
- Multi-turn conversation view
- Answers only from your document, not hallucinated

### 📊 Scores Tab
- 5-dimension scoring: Clarity, Feasibility, Value Prop, Structure, Overall
- Color-coded (green/yellow/red)
- One-line AI verdict

---

## Project Structure

```
proposal_rag/
├── main.py              # Entry point
├── setup.py             # One-time setup
├── requirements.txt     # Python dependencies
├── core/
│   ├── rag_pipeline.py  # Document loading → chunking → embedding → ChromaDB
│   ├── llm_client.py    # Ollama client + prompt templates
│   └── workers.py       # QThread workers (non-blocking async)
└── ui/
    └── main_window.py   # Full PyQt6 application UI
```

---

## Configuration (in the UI sidebar)

| Setting          | Default            | Notes                          |
|------------------|--------------------|--------------------------------|
| LLM Model        | llama3.2           | Any locally pulled Ollama model|
| Embedding Model  | nomic-embed-text   | Or mxbai-embed-large           |
| Chunk Size       | 512 tokens         | Adjust 128–1024 via slider     |

---

## Adding More Models

```bash
ollama pull mistral
ollama pull phi3
ollama pull mxbai-embed-large
```

Then click **⟳ Refresh Models** in the app sidebar.

---

## Offline Operation

All components run entirely offline:
- Ollama runs models locally (no API calls)
- ChromaDB persists to `~/.proposal_rag/chroma_db`
- No internet connection required after initial model download
