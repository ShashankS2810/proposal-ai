"""
Proposal Review & Improvement System
Powered by Ollama + RAG (LangChain + ChromaDB)
"""

import sys
from PyQt6.QtWidgets import QApplication
from ui.main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("ProposalAI")
    app.setOrganizationName("ProposalReview")
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
