"""
RAG Pipeline Core
Handles: Document loading → Chunking → Embeddings → ChromaDB → Retrieval
"""

import os
import hashlib
from pathlib import Path
from typing import List, Optional, Callable

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    Docx2txtLoader,
)
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document

CHROMA_PERSIST_DIR = str(Path.home() / ".proposal_rag" / "chroma_db")
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".csv"}


class RAGPipeline:
    """
    Full RAG pipeline:
      1. Load documents from file
      2. Split into chunks
      3. Embed with nomic-embed-text via Ollama
      4. Store/retrieve from ChromaDB
    """

    def __init__(
        self,
        embedding_model: str = "nomic-embed-text",
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        progress_callback: Optional[Callable[[str, int], None]] = None,
    ):
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.progress_callback = progress_callback  # fn(message, percent)

        self.embeddings = OllamaEmbeddings(model=embedding_model)
        self.vectorstore: Optional[Chroma] = None
        self.current_collection: Optional[str] = None

        os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)

    def _emit(self, msg: str, pct: int):
        if self.progress_callback:
            self.progress_callback(msg, pct)

    # ── Document Loading ──────────────────────────────────────────────────────

    def load_document(self, file_path: str) -> List[Document]:
        ext = Path(file_path).suffix.lower()
        self._emit(f"Loading {Path(file_path).name}…", 10)

        if ext == ".pdf":
            loader = PyPDFLoader(file_path)
        elif ext == ".docx":
            loader = Docx2txtLoader(file_path)
        elif ext == ".txt":
            loader = TextLoader(file_path, encoding="utf-8")
        elif ext == ".csv":
            loader = CSVLoader(file_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")

        docs = loader.load()
        self._emit(f"Loaded {len(docs)} page(s)", 25)
        return docs

    # ── Text Splitting ────────────────────────────────────────────────────────

    def split_documents(self, docs: List[Document]) -> List[Document]:
        self._emit("Chunking text…", 35)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks = splitter.split_documents(docs)
        self._emit(f"Created {len(chunks)} chunks", 50)
        return chunks

    # ── Vector Store ──────────────────────────────────────────────────────────

    def _collection_name(self, file_path: str) -> str:
        """Stable collection name from file path hash."""
        h = hashlib.md5(file_path.encode()).hexdigest()[:8]
        stem = Path(file_path).stem[:20].replace(" ", "_")
        return f"{stem}_{h}"

    def ingest(self, file_path: str) -> str:
        """Full ingestion pipeline. Returns collection name."""
        docs = self.load_document(file_path)
        chunks = self.split_documents(docs)

        collection = self._collection_name(file_path)
        self._emit("Generating embeddings…", 60)

        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            collection_name=collection,
            persist_directory=CHROMA_PERSIST_DIR,
        )
        self.current_collection = collection
        self._emit("Stored in vector DB ✓", 90)
        return collection

    def load_existing(self, collection_name: str):
        """Load a previously persisted collection."""
        self.vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=CHROMA_PERSIST_DIR,
        )
        self.current_collection = collection_name

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def retrieve(self, query: str, k: int = 6) -> List[Document]:
        if not self.vectorstore:
            raise RuntimeError("No document loaded. Ingest a file first.")
        retriever = self.vectorstore.as_retriever(
            search_type="mmr",           # Maximal Marginal Relevance — diverse chunks
            search_kwargs={"k": k, "fetch_k": k * 3},
        )
        return retriever.invoke(query)

    def get_context(self, query: str, k: int = 6) -> str:
        """Return retrieved chunks joined as a single context string."""
        docs = self.retrieve(query, k=k)
        return "\n\n---\n\n".join(d.page_content for d in docs)
