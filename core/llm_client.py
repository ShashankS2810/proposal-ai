"""
Ollama LLM Interface
Handles: Review, Score, Improve, Q&A — all using retrieved RAG context
"""

from typing import Generator, Optional
import ollama


# ── System Prompts ────────────────────────────────────────────────────────────

REVIEW_SYSTEM = """You are an expert proposal evaluator and business strategist.
You review proposals thoroughly and provide structured, actionable feedback.
Base your review ONLY on the provided document context.
Be honest, specific, and constructive."""

IMPROVE_SYSTEM = """You are a senior proposal writer and strategist.
Your job is to rewrite and improve proposals to be more compelling, clear, and persuasive.
Use ONLY the content from the provided document context.
Preserve the author's intent while significantly improving quality."""

QA_SYSTEM = """You are a helpful assistant that answers questions about proposals.
Answer ONLY using the provided document context.
If the answer is not in the context, say so clearly."""


# ── Prompt Templates ──────────────────────────────────────────────────────────

def build_review_prompt(context: str) -> str:
    return f"""Here is the proposal content:

{context}

---

Please provide a comprehensive review in the following structure:

## 📊 Scores (out of 10)
- **Clarity**: X/10
- **Feasibility**: X/10
- **Value Proposition**: X/10
- **Structure & Flow**: X/10
- **Overall**: X/10

## ✅ Strengths
(List 3-5 key strengths)

## ⚠️ Weaknesses
(List 3-5 key weaknesses)

## 🎯 Recommendations
(List 5+ specific, actionable improvements)

## 💬 Summary
(2-3 sentence executive summary of your review)
"""


def build_improvement_prompt(context: str, focus: Optional[str] = None) -> str:
    focus_line = f"\nFocus especially on improving: {focus}\n" if focus else ""
    return f"""Here is the original proposal content:

{context}
{focus_line}
---

Please produce an improved version of this proposal that:
1. Has a stronger opening hook
2. Clearly states the problem and solution
3. Highlights the value proposition
4. Uses persuasive, professional language
5. Has a clear call to action

Write the full improved proposal below:
"""


def build_qa_prompt(context: str, question: str) -> str:
    return f"""Document context:

{context}

---

Question: {question}

Answer based only on the above context:"""


def build_score_prompt(context: str) -> str:
    return f"""Here is the proposal content:

{context}

---

Evaluate this proposal and return ONLY a JSON object with these exact keys:
{{
  "clarity": <0-10>,
  "feasibility": <0-10>,
  "value_proposition": <0-10>,
  "structure": <0-10>,
  "overall": <0-10>,
  "one_line_verdict": "<15 words or less>"
}}
No extra text, just valid JSON."""


# ── LLM Client ────────────────────────────────────────────────────────────────

class OllamaClient:
    def __init__(self, model: str = "llama3.2"):
        self.model = model

    def _chat(self, system: str, user: str, stream: bool = False):
        messages = [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ]
        if stream:
            return ollama.chat(model=self.model, messages=messages, stream=True)
        else:
            resp = ollama.chat(model=self.model, messages=messages)
            return resp["message"]["content"]

    # ── Public API ────────────────────────────────────────────────────────────

    def review(self, context: str, stream: bool = True):
        """Review proposal. Returns string or generator."""
        prompt = build_review_prompt(context)
        return self._chat(REVIEW_SYSTEM, prompt, stream=stream)

    def improve(self, context: str, focus: Optional[str] = None, stream: bool = True):
        """Improve/rewrite proposal. Returns string or generator."""
        prompt = build_improvement_prompt(context, focus)
        return self._chat(IMPROVE_SYSTEM, prompt, stream=stream)

    def answer(self, context: str, question: str, stream: bool = True):
        """Answer a question about the proposal."""
        prompt = build_qa_prompt(context, question)
        return self._chat(QA_SYSTEM, prompt, stream=stream)

    def score(self, context: str) -> dict:
        """Return JSON scores dict."""
        import json
        prompt = build_score_prompt(context)
        raw = self._chat(REVIEW_SYSTEM, prompt, stream=False)
        try:
            # Strip markdown fences if present
            clean = raw.strip().strip("```json").strip("```").strip()
            return json.loads(clean)
        except Exception:
            return {
                "clarity": 0, "feasibility": 0, "value_proposition": 0,
                "structure": 0, "overall": 0, "one_line_verdict": "Parse error"
            }

    def list_models(self) -> list[str]:
        """List locally available Ollama models."""
        try:
            result = ollama.list()
            return [m["name"] for m in result.get("models", [])]
        except Exception:
            return ["llama3.2", "mistral", "phi3"]

    def is_available(self) -> bool:
        try:
            ollama.list()
            return True
        except Exception:
            return False
