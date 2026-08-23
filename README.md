# The Grounded Answer

> **CLI RAG Assistant with Date-Aware Clause Resolution & Grounded Refusal Gates**

"The Grounded Answer" answers policy questions from an official manual, cites exact clauses (e.g., `§4.3.2`), resolves date-dependent amendments with clause-specific transitional rules, and strictly refuses to answer when out of scope or contradictory.

---

## 🏗️ Architecture & Separable Modules

This project is built with strictly separable components:

| Module | Responsibility |
|---|---|
| [`ingest.py`](file:///./ingest.py) | Parses corpus markdown by `§x.x.x` clause boundaries, extracts metadata, and indexes in local/in-memory Qdrant vector store. |
| [`clause_resolver.py`](file:///./clause_resolver.py) | Standalone engine that evaluates claim dates against explicit per-clause `TransitionalRule`s to return the legally correct text. |
| [`retriever.py`](file:///./retriever.py) | Queries in-memory Qdrant for top-$k$ relevant clauses. |
| [`refusal_gate.py`](file:///./refusal_gate.py) | Pre-generation gate evaluating retrieval confidence, topic scope, and contradictions before allowing generation. |
| [`answer_builder.py`](file:///./answer_builder.py) | Synthesizes answers using Gemini 2.5 Flash with strict clause-level citation traceability. |
| [`cli.py`](file:///./cli.py) | CLI interface (one question in, one grounded answer out, with claim date prompting/flag). |
| [`main.py`](file:///./main.py) | End-to-end orchestrator wiring all modules. |

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.11+
- Git

### 2. Installation & Virtual Environment

```bash
# Clone the repository (or navigate to workspace)
cd the-grounded-answer

# Create and activate virtual environment
python -m venv venv
# On Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# On macOS/Linux:
# source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure Environment Variables
Copy `.env.example` to `.env` and provide your Google Gemini API key:
```bash
cp .env.example .env
```
Inside `.env`:
```env
GOOGLE_API_KEY=your_gemini_api_key_here
```

### 4. Running the CLI
Run with an inline claim date flag:
```bash
python cli.py "What is the lodging reimbursement limit?" -d 2026-03-15
```

Or run interactively (you will be prompted for question and claim date):
```bash
python cli.py
```

### 5. Running Tests
Run automated test suite via `pytest`:
```bash
pytest
```
