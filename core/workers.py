"""
Qt Worker Threads
Runs RAG ingestion and LLM calls off the main thread to keep UI responsive.
"""

from PyQt6.QtCore import QThread, pyqtSignal


class IngestWorker(QThread):
    """Ingest a document into the vector store."""

    progress = pyqtSignal(str, int)   # (message, percent)
    finished = pyqtSignal(str)         # collection_name
    error    = pyqtSignal(str)

    def __init__(self, rag, file_path: str):
        super().__init__()
        self.rag = rag
        self.file_path = file_path

    def run(self):
        try:
            self.rag.progress_callback = lambda msg, pct: self.progress.emit(msg, pct)
            collection = self.rag.ingest(self.file_path)
            self.finished.emit(collection)
        except Exception as e:
            self.error.emit(str(e))


class LLMWorker(QThread):
    """Run an LLM call and stream tokens back."""

    token    = pyqtSignal(str)   # streamed token
    finished = pyqtSignal()
    error    = pyqtSignal(str)

    def __init__(self, llm_fn):
        """
        llm_fn: callable that returns an ollama stream generator
        """
        super().__init__()
        self.llm_fn = llm_fn

    def run(self):
        try:
            stream = self.llm_fn()
            for chunk in stream:
                content = chunk["message"]["content"]
                if content:
                    self.token.emit(content)
            self.finished.emit()
        except Exception as e:
            self.error.emit(str(e))


class ScoreWorker(QThread):
    """Fetch JSON scores from LLM."""

    result   = pyqtSignal(dict)
    error    = pyqtSignal(str)

    def __init__(self, llm_client, context: str):
        super().__init__()
        self.llm_client = llm_client
        self.context = context

    def run(self):
        try:
            scores = self.llm_client.score(self.context)
            self.result.emit(scores)
        except Exception as e:
            self.error.emit(str(e))
