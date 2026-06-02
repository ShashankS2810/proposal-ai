#!/usr/bin/env python3
"""
Setup script — installs all dependencies and pulls required Ollama models.
Run once before launching the app.
"""

import subprocess
import sys


def run(cmd: list[str], desc: str):
    print(f"\n{'─'*50}")
    print(f"  {desc}")
    print(f"{'─'*50}")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"  ⚠️  Warning: command returned {result.returncode}")
    return result.returncode


def main():
    print("\n" + "═"*50)
    print("  ProposalAI — Setup")
    print("═"*50)

    # 1. Install Python deps
    run(
        [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
        "Installing Python dependencies…"
    )

    # 2. Pull Ollama models
    print("\n  Pulling Ollama models (this may take a while)…")
    for model in ["nomic-embed-text", "llama3.2"]:
        run(["ollama", "pull", model], f"Pulling {model}…")

    print("\n" + "═"*50)
    print("  ✅ Setup complete!")
    print("  Run:  python main.py")
    print("═"*50 + "\n")


if __name__ == "__main__":
    main()
