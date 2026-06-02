"""
Main Application Window
Professional PyQt6 UI for the Proposal Review & Improvement System
"""

import os
from pathlib import Path

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QPushButton, QLabel, QFileDialog, QTextEdit, QLineEdit,
    QComboBox, QProgressBar, QFrame, QTabWidget, QScrollArea,
    QGroupBox, QSlider, QMessageBox, QSizePolicy,
)
from PyQt6.QtCore import Qt, QSize, pyqtSlot
from PyQt6.QtGui import QFont, QColor, QPalette, QIcon, QPixmap, QTextCursor

from core.rag_pipeline import RAGPipeline, SUPPORTED_EXTENSIONS
from core.llm_client import OllamaClient
from core.workers import IngestWorker, LLMWorker, ScoreWorker


# ── Stylesheet ────────────────────────────────────────────────────────────────

STYLE = """
QMainWindow, QWidget {
    background-color: #0f1117;
    color: #e2e8f0;
    font-family: 'Helvetica', 'SF Pro Display', sans-serif;
}

/* Sidebar */
#sidebar {
    background-color: #161b27;
    border-right: 1px solid #1e2a3a;
    min-width: 240px;
    max-width: 280px;
}

#appTitle {
    font-size: 18px;
    font-weight: 700;
    color: #7dd3fc;
    letter-spacing: 1px;
    padding: 8px 0;
}

#appSubtitle {
    font-size: 11px;
    color: #64748b;
    letter-spacing: 2px;
    text-transform: uppercase;
}

/* File Drop Zone */
#dropZone {
    background-color: #1a2233;
    border: 2px dashed #2d4a6e;
    border-radius: 10px;
    padding: 20px;
    color: #64748b;
    font-size: 13px;
    text-align: center;
}

#dropZone:hover {
    border-color: #7dd3fc;
    color: #7dd3fc;
    background-color: #1e2d44;
}

/* Buttons */
QPushButton {
    background-color: #1e3a5f;
    color: #93c5fd;
    border: 1px solid #2d5a8e;
    border-radius: 6px;
    padding: 8px 16px;
    font-size: 13px;
    font-weight: 500;
}

QPushButton:hover {
    background-color: #2d4f7c;
    border-color: #7dd3fc;
    color: #e0f2fe;
}

QPushButton:pressed {
    background-color: #1a3555;
}

QPushButton:disabled {
    background-color: #1a2233;
    color: #374151;
    border-color: #1e2a3a;
}

#primaryBtn {
    background-color: #0369a1;
    color: #e0f2fe;
    border: none;
    font-weight: 600;
    padding: 10px 20px;
    font-size: 14px;
}

#primaryBtn:hover {
    background-color: #0284c7;
}

#dangerBtn {
    background-color: #7f1d1d;
    color: #fca5a5;
    border: 1px solid #991b1b;
}

#dangerBtn:hover {
    background-color: #991b1b;
}

/* Tabs */
QTabWidget::pane {
    border: 1px solid #1e2a3a;
    border-radius: 8px;
    background: #0f1117;
}

QTabBar::tab {
    background: #161b27;
    color: #64748b;
    padding: 10px 20px;
    border: none;
    font-size: 13px;
    font-weight: 500;
}

QTabBar::tab:selected {
    background: #1e3a5f;
    color: #7dd3fc;
    border-bottom: 2px solid #7dd3fc;
}

QTabBar::tab:hover:!selected {
    background: #1a2233;
    color: #93c5fd;
}

/* Text areas */
QTextEdit {
    background-color: #131920;
    color: #d1d5db;
    border: 1px solid #1e2a3a;
    border-radius: 8px;
    padding: 12px;
    font-size: 13px;
    line-height: 1.6;
    font-family: 'Cascadia Code', 'Fira Code', 'Consolas', monospace;
    selection-background-color: #1e3a5f;
}

/* Input */
QLineEdit {
    background-color: #161b27;
    color: #d1d5db;
    border: 1px solid #1e2a3a;
    border-radius: 6px;
    padding: 8px 12px;
    font-size: 13px;
}

QLineEdit:focus {
    border-color: #7dd3fc;
    background-color: #1a2233;
}

/* Combo */
QComboBox {
    background-color: #161b27;
    color: #d1d5db;
    border: 1px solid #1e2a3a;
    border-radius: 6px;
    padding: 6px 12px;
    font-size: 13px;
    min-width: 160px;
}

QComboBox::drop-down {
    border: none;
    padding-right: 8px;
}

QComboBox QAbstractItemView {
    background-color: #161b27;
    color: #d1d5db;
    selection-background-color: #1e3a5f;
    border: 1px solid #1e2a3a;
}

/* Progress bar */
QProgressBar {
    background-color: #1a2233;
    border: none;
    border-radius: 4px;
    height: 6px;
    text-align: center;
    font-size: 11px;
    color: transparent;
}

QProgressBar::chunk {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #0369a1, stop:1 #7dd3fc);
    border-radius: 4px;
}

/* Score cards */
#scoreCard {
    background-color: #161b27;
    border: 1px solid #1e2a3a;
    border-radius: 8px;
    padding: 12px;
    margin: 4px;
}

#scoreLabel {
    color: #64748b;
    font-size: 11px;
    letter-spacing: 1px;
    text-transform: uppercase;
}

#scoreValue {
    color: #7dd3fc;
    font-size: 28px;
    font-weight: 700;
}

/* Status bar */
#statusBar {
    background-color: #161b27;
    border-top: 1px solid #1e2a3a;
    padding: 4px 12px;
    font-size: 12px;
    color: #64748b;
}

#statusDot {
    font-size: 10px;
}

/* Group boxes */
QGroupBox {
    border: 1px solid #1e2a3a;
    border-radius: 8px;
    margin-top: 12px;
    padding: 8px;
    color: #64748b;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 1px;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 6px;
    color: #64748b;
    text-transform: uppercase;
}

/* Scrollbar */
QScrollBar:vertical {
    background: #0f1117;
    width: 6px;
    border: none;
}
QScrollBar::handle:vertical {
    background: #1e2a3a;
    border-radius: 3px;
    min-height: 20px;
}
QScrollBar::handle:vertical:hover {
    background: #2d4a6e;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }

/* Separator */
QFrame[frameShape="4"], QFrame[frameShape="5"] {
    color: #1e2a3a;
}
"""


# ── Score Widget ──────────────────────────────────────────────────────────────

class ScoreCard(QFrame):
    def __init__(self, label: str, parent=None):
        super().__init__(parent)
        self.setObjectName("scoreCard")
        layout = QVBoxLayout(self)
        layout.setSpacing(4)

        self.value_lbl = QLabel("—")
        self.value_lbl.setObjectName("scoreValue")
        self.value_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.label_lbl = QLabel(label)
        self.label_lbl.setObjectName("scoreLabel")
        self.label_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout.addWidget(self.value_lbl)
        layout.addWidget(self.label_lbl)

    def set_score(self, value: int):
        self.value_lbl.setText(f"{value}/10")
        color = "#4ade80" if value >= 7 else "#fbbf24" if value >= 5 else "#f87171"
        self.value_lbl.setStyleSheet(f"color: {color}; font-size: 28px; font-weight: 700;")


# ── Main Window ───────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ProposalAI — Review & Improvement System")
        self.setMinimumSize(1200, 750)
        self.resize(1400, 850)

        self.rag = RAGPipeline()
        self.llm = OllamaClient()
        self.current_file: str | None = None
        self.ingest_worker: IngestWorker | None = None
        self.llm_worker: LLMWorker | None = None

        self.setStyleSheet(STYLE)
        self._build_ui()
        self._check_ollama()
        self._refresh_models()

    # ── UI Construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Sidebar
        sidebar = self._build_sidebar()
        root.addWidget(sidebar)

        # Main content
        main_content = self._build_main_content()
        root.addWidget(main_content, stretch=1)

    def _build_sidebar(self) -> QWidget:
        sidebar = QWidget()
        sidebar.setObjectName("sidebar")
        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(16, 20, 16, 16)
        layout.setSpacing(12)

        # Title
        title = QLabel("ProposalAI")
        title.setObjectName("appTitle")
        subtitle = QLabel("RAG · OLLAMA · LOCAL")
        subtitle.setObjectName("appSubtitle")
        layout.addWidget(title)
        layout.addWidget(subtitle)

        layout.addWidget(self._separator())

        # Model selector
        model_group = QGroupBox("LLM Model")
        mg_layout = QVBoxLayout(model_group)
        self.model_combo = QComboBox()
        self.model_combo.addItem("llama3.2")
        self.model_combo.currentTextChanged.connect(self._on_model_changed)
        mg_layout.addWidget(self.model_combo)

        refresh_btn = QPushButton("⟳ Refresh Models")
        refresh_btn.clicked.connect(self._refresh_models)
        mg_layout.addWidget(refresh_btn)
        layout.addWidget(model_group)

        # Embedding selector
        embed_group = QGroupBox("Embedding Model")
        eg_layout = QVBoxLayout(embed_group)
        self.embed_combo = QComboBox()
        self.embed_combo.addItems(["nomic-embed-text", "mxbai-embed-large"])
        eg_layout.addWidget(self.embed_combo)
        layout.addWidget(embed_group)

        # Chunk settings
        chunk_group = QGroupBox("Chunk Size")
        cg_layout = QVBoxLayout(chunk_group)
        self.chunk_slider = QSlider(Qt.Orientation.Horizontal)
        self.chunk_slider.setRange(128, 1024)
        self.chunk_slider.setValue(512)
        self.chunk_slider.setTickInterval(128)
        self.chunk_size_lbl = QLabel("512 tokens")
        self.chunk_size_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.chunk_size_lbl.setStyleSheet("color: #7dd3fc; font-size: 12px;")
        self.chunk_slider.valueChanged.connect(
            lambda v: self.chunk_size_lbl.setText(f"{v} tokens")
        )
        cg_layout.addWidget(self.chunk_slider)
        cg_layout.addWidget(self.chunk_size_lbl)
        layout.addWidget(chunk_group)

        layout.addWidget(self._separator())

        # Ollama status
        self.ollama_status = QLabel("● Checking Ollama…")
        self.ollama_status.setObjectName("statusDot")
        self.ollama_status.setStyleSheet("color: #fbbf24; font-size: 12px;")
        layout.addWidget(self.ollama_status)

        layout.addStretch()

        # Clear button
        clear_btn = QPushButton("🗑 Clear Session")
        clear_btn.setObjectName("dangerBtn")
        clear_btn.clicked.connect(self._clear_session)
        layout.addWidget(clear_btn)

        return sidebar

    def _build_main_content(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Top bar
        topbar = self._build_topbar()
        layout.addWidget(topbar)

        # Progress
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setFixedHeight(6)
        layout.addWidget(self.progress_bar)

        # Tabs
        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_review_tab(), "📋 Review")
        self.tabs.addTab(self._build_improve_tab(), "✨ Improve")
        self.tabs.addTab(self._build_qa_tab(), "💬 Ask Questions")
        self.tabs.addTab(self._build_scores_tab(), "📊 Scores")
        layout.addWidget(self.tabs, stretch=1)

        # Status bar
        statusbar = self._build_statusbar()
        layout.addWidget(statusbar)

        return widget

    def _build_topbar(self) -> QWidget:
        bar = QFrame()
        bar.setStyleSheet("background:#161b27; border-bottom:1px solid #1e2a3a;")
        bar.setFixedHeight(60)
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(16, 8, 16, 8)

        self.file_label = QLabel("No document loaded")
        self.file_label.setStyleSheet("color:#64748b; font-size:13px;")

        self.load_btn = QPushButton("📂 Load Document")
        self.load_btn.setObjectName("primaryBtn")
        self.load_btn.clicked.connect(self._load_file)

        self.ingest_btn = QPushButton("⚡ Ingest into RAG")
        self.ingest_btn.setEnabled(False)
        self.ingest_btn.clicked.connect(self._start_ingest)

        layout.addWidget(self.load_btn)
        layout.addWidget(self.ingest_btn)
        layout.addWidget(QLabel("|"), 0)
        layout.addWidget(self.file_label, stretch=1)

        return bar

    def _build_review_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        info = QLabel(
            "AI reviews your proposal and provides structured feedback with strengths, "
            "weaknesses, and actionable recommendations."
        )
        info.setStyleSheet("color:#64748b; font-size:12px;")
        info.setWordWrap(True)
        layout.addWidget(info)

        self.review_btn = QPushButton("🔍 Generate Review")
        self.review_btn.setObjectName("primaryBtn")
        self.review_btn.setEnabled(False)
        self.review_btn.clicked.connect(self._run_review)

        self.stop_review_btn = QPushButton("⏹ Stop")
        self.stop_review_btn.setEnabled(False)
        self.stop_review_btn.clicked.connect(self._stop_llm)

        btn_row = QHBoxLayout()
        btn_row.addWidget(self.review_btn)
        btn_row.addWidget(self.stop_review_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        self.review_output = QTextEdit()
        self.review_output.setReadOnly(True)
        self.review_output.setPlaceholderText(
            "Review output will appear here…\n\nLoad and ingest a document first, then click Generate Review."
        )
        layout.addWidget(self.review_output, stretch=1)
        return widget

    def _build_improve_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        info = QLabel(
            "AI rewrites your proposal to be more compelling, clear, and persuasive while preserving your intent."
        )
        info.setStyleSheet("color:#64748b; font-size:12px;")
        info.setWordWrap(True)
        layout.addWidget(info)

        focus_row = QHBoxLayout()
        focus_lbl = QLabel("Focus on (optional):")
        focus_lbl.setStyleSheet("color:#94a3b8; font-size:12px;")
        self.focus_input = QLineEdit()
        self.focus_input.setPlaceholderText(
            "e.g. executive summary, budget justification, problem statement…"
        )
        focus_row.addWidget(focus_lbl)
        focus_row.addWidget(self.focus_input, stretch=1)
        layout.addLayout(focus_row)

        self.improve_btn = QPushButton("✨ Improve Proposal")
        self.improve_btn.setObjectName("primaryBtn")
        self.improve_btn.setEnabled(False)
        self.improve_btn.clicked.connect(self._run_improve)

        self.stop_improve_btn = QPushButton("⏹ Stop")
        self.stop_improve_btn.setEnabled(False)
        self.stop_improve_btn.clicked.connect(self._stop_llm)

        btn_row = QHBoxLayout()
        btn_row.addWidget(self.improve_btn)
        btn_row.addWidget(self.stop_improve_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        self.improve_output = QTextEdit()
        self.improve_output.setReadOnly(True)
        self.improve_output.setPlaceholderText(
            "Improved proposal will appear here…\n\nLoad and ingest a document first, then click Improve Proposal."
        )
        layout.addWidget(self.improve_output, stretch=1)
        return widget

    def _build_qa_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        info = QLabel(
            "Ask specific questions about the proposal. Answers are grounded in the document via RAG."
        )
        info.setStyleSheet("color:#64748b; font-size:12px;")
        info.setWordWrap(True)
        layout.addWidget(info)

        self.qa_output = QTextEdit()
        self.qa_output.setReadOnly(True)
        self.qa_output.setPlaceholderText(
            "Conversation will appear here…\n\nLoad and ingest a document, then ask a question below."
        )
        layout.addWidget(self.qa_output, stretch=1)

        qa_row = QHBoxLayout()
        self.qa_input = QLineEdit()
        self.qa_input.setPlaceholderText("Ask anything about this proposal…")
        self.qa_input.returnPressed.connect(self._run_qa)

        self.qa_btn = QPushButton("Ask ↵")
        self.qa_btn.setObjectName("primaryBtn")
        self.qa_btn.setEnabled(False)
        self.qa_btn.clicked.connect(self._run_qa)

        qa_row.addWidget(self.qa_input, stretch=1)
        qa_row.addWidget(self.qa_btn)
        layout.addLayout(qa_row)
        return widget

    def _build_scores_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(16)

        info = QLabel(
            "Quantitative scores across 5 dimensions, powered by Ollama's analysis of your proposal."
        )
        info.setStyleSheet("color:#64748b; font-size:12px;")
        info.setWordWrap(True)
        layout.addWidget(info)

        self.score_btn = QPushButton("📊 Generate Scores")
        self.score_btn.setObjectName("primaryBtn")
        self.score_btn.setEnabled(False)
        self.score_btn.clicked.connect(self._run_scores)
        layout.addWidget(self.score_btn, alignment=Qt.AlignmentFlag.AlignLeft)

        # Score cards row
        cards_row = QHBoxLayout()
        self.score_cards = {}
        for key, label in [
            ("clarity",           "Clarity"),
            ("feasibility",       "Feasibility"),
            ("value_proposition", "Value Prop"),
            ("structure",         "Structure"),
            ("overall",           "Overall"),
        ]:
            card = ScoreCard(label)
            self.score_cards[key] = card
            cards_row.addWidget(card)
        layout.addLayout(cards_row)

        self.verdict_lbl = QLabel("—")
        self.verdict_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.verdict_lbl.setStyleSheet("color:#7dd3fc; font-size:15px; font-style:italic; padding:12px;")
        layout.addWidget(self.verdict_lbl)

        layout.addStretch()
        return widget

    def _build_statusbar(self) -> QWidget:
        bar = QFrame()
        bar.setObjectName("statusBar")
        bar.setFixedHeight(28)
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(12, 0, 12, 0)

        self.status_lbl = QLabel("Ready — load a document to begin")
        self.status_lbl.setStyleSheet("color:#64748b; font-size:12px;")
        layout.addWidget(self.status_lbl)
        layout.addStretch()

        self.model_status_lbl = QLabel(f"Model: {self.llm.model}")
        self.model_status_lbl.setStyleSheet("color:#475569; font-size:12px;")
        layout.addWidget(self.model_status_lbl)
        return bar

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _separator() -> QFrame:
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Plain)
        return line

    def _set_status(self, msg: str):
        self.status_lbl.setText(msg)

    def _enable_actions(self, enabled: bool):
        self.review_btn.setEnabled(enabled)
        self.improve_btn.setEnabled(enabled)
        self.qa_btn.setEnabled(enabled)
        self.score_btn.setEnabled(enabled)

    # ── Slots ─────────────────────────────────────────────────────────────────

    def _check_ollama(self):
        if self.llm.is_available():
            self.ollama_status.setText("● Ollama running")
            self.ollama_status.setStyleSheet("color: #4ade80; font-size: 12px;")
        else:
            self.ollama_status.setText("● Ollama offline")
            self.ollama_status.setStyleSheet("color: #f87171; font-size: 12px;")

    def _refresh_models(self):
        models = self.llm.list_models()
        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        self.model_combo.addItems(models if models else ["llama3.2"])
        self.model_combo.blockSignals(False)
        self._check_ollama()

    def _on_model_changed(self, model: str):
        self.llm.model = model
        self.model_status_lbl.setText(f"Model: {model}")

    def _load_file(self):
        exts = " ".join(f"*{e}" for e in SUPPORTED_EXTENSIONS)
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Proposal Document", "",
            f"Documents ({exts});;All Files (*)"
        )
        if not path:
            return
        self.current_file = path
        self.file_label.setText(f"📄 {Path(path).name}")
        self.ingest_btn.setEnabled(True)
        self._set_status(f"Loaded: {Path(path).name}  — click ⚡ Ingest to process")
        self._enable_actions(False)

    def _start_ingest(self):
        if not self.current_file:
            return

        # Update RAG settings from sidebar
        self.rag.embedding_model = self.embed_combo.currentText()
        self.rag.embeddings = __import__(
            "langchain_ollama", fromlist=["OllamaEmbeddings"]
        ).OllamaEmbeddings(model=self.rag.embedding_model)
        self.rag.chunk_size = self.chunk_slider.value()

        self.ingest_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self._set_status("Ingesting document…")

        self.ingest_worker = IngestWorker(self.rag, self.current_file)
        self.ingest_worker.progress.connect(self._on_ingest_progress)
        self.ingest_worker.finished.connect(self._on_ingest_done)
        self.ingest_worker.error.connect(self._on_error)
        self.ingest_worker.start()

    @pyqtSlot(str, int)
    def _on_ingest_progress(self, msg: str, pct: int):
        self.progress_bar.setValue(pct)
        self._set_status(msg)

    @pyqtSlot(str)
    def _on_ingest_done(self, collection: str):
        self.progress_bar.setValue(100)
        self.progress_bar.setVisible(False)
        self._set_status(f"✅ Document ingested → collection: {collection}")
        self._enable_actions(True)
        self.ingest_btn.setEnabled(True)

    @pyqtSlot(str)
    def _on_error(self, err: str):
        self.progress_bar.setVisible(False)
        self._set_status(f"❌ Error: {err}")
        QMessageBox.critical(self, "Error", err)

    # ── LLM Actions ───────────────────────────────────────────────────────────

    def _run_review(self):
        context = self.rag.get_context(
            "proposal overview objectives problem solution value proposition budget"
        )
        self.review_output.clear()
        self._stream_to(
            lambda: self.llm.review(context, stream=True),
            self.review_output,
            self.review_btn,
            self.stop_review_btn,
        )

    def _run_improve(self):
        focus = self.focus_input.text().strip() or None
        context = self.rag.get_context(
            "proposal full content objectives solution implementation budget"
        )
        self.improve_output.clear()
        self._stream_to(
            lambda: self.llm.improve(context, focus=focus, stream=True),
            self.improve_output,
            self.improve_btn,
            self.stop_improve_btn,
        )

    def _run_qa(self):
        question = self.qa_input.text().strip()
        if not question:
            return
        context = self.rag.get_context(question)
        self.qa_output.append(f"\n🧑 **You:** {question}\n")
        self.qa_input.clear()
        self.qa_output.append("🤖 **Assistant:** ")
        self._stream_to(
            lambda: self.llm.answer(context, question, stream=True),
            self.qa_output,
            self.qa_btn,
            None,
            append_mode=True,
        )

    def _run_scores(self):
        context = self.rag.get_context(
            "proposal scores clarity feasibility value structure"
        )
        self.score_btn.setEnabled(False)
        self._set_status("Scoring proposal…")
        self.score_worker = ScoreWorker(self.llm, context)
        self.score_worker.result.connect(self._on_scores)
        self.score_worker.error.connect(self._on_error)
        self.score_worker.start()

    @pyqtSlot(dict)
    def _on_scores(self, scores: dict):
        for key, card in self.score_cards.items():
            card.set_score(int(scores.get(key, 0)))
        self.verdict_lbl.setText(f'"{scores.get("one_line_verdict", "")}"')
        self.score_btn.setEnabled(True)
        self._set_status("✅ Scores generated")

    def _stream_to(self, llm_fn, output: QTextEdit, run_btn: QPushButton,
                   stop_btn: QPushButton | None, append_mode: bool = False):
        run_btn.setEnabled(False)
        if stop_btn:
            stop_btn.setEnabled(True)
        self._set_status("Generating…")

        self.llm_worker = LLMWorker(llm_fn)
        self.llm_worker.token.connect(
            lambda t: self._append_token(t, output, append_mode)
        )
        self.llm_worker.finished.connect(
            lambda: self._on_stream_done(run_btn, stop_btn)
        )
        self.llm_worker.error.connect(self._on_error)
        self.llm_worker.start()

    def _append_token(self, token: str, output: QTextEdit, append_mode: bool):
        cursor = output.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertText(token)
        output.setTextCursor(cursor)
        output.ensureCursorVisible()

    def _on_stream_done(self, run_btn: QPushButton, stop_btn: QPushButton | None):
        run_btn.setEnabled(True)
        if stop_btn:
            stop_btn.setEnabled(False)
        self._set_status("✅ Done")

    def _stop_llm(self):
        if self.llm_worker and self.llm_worker.isRunning():
            self.llm_worker.terminate()
            self._set_status("⏹ Stopped")

    def _clear_session(self):
        reply = QMessageBox.question(
            self, "Clear Session",
            "Clear the current session? The vector store for this document will be removed.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.rag.vectorstore = None
            self.current_file = None
            self.file_label.setText("No document loaded")
            self.ingest_btn.setEnabled(False)
            self._enable_actions(False)
            self.review_output.clear()
            self.improve_output.clear()
            self.qa_output.clear()
            for card in self.score_cards.values():
                card.value_lbl.setText("—")
                card.value_lbl.setStyleSheet("color:#7dd3fc; font-size:28px; font-weight:700;")
            self.verdict_lbl.setText("—")
            self._set_status("Session cleared")
