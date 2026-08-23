# The Grounded Answer

> **CLI RAG Assistant with Date-Aware Clause Resolution & Grounded Refusal Gates**

"The Grounded Answer" is a CLI assistant built for policy inquiries. It answers questions directly from an official manual, cites exact clauses (e.g., `§6.4.1(a)`), resolves date-dependent amendments with clause-specific transitional rules, and strictly refuses to answer when out of scope, low-confidence, or contradictory.

---

## 🏗️ Architecture & Separable Modules

| Module | Responsibility |
|---|---|
| [`ingest.py`](file:///./ingest.py) | Parses corpus markdown by `§P.S.p` clause boundaries, extracts metadata, links amendment provisions to Part 5 transitional rules, and indexes vectors in local Qdrant. |
| [`clause_resolver.py`](file:///./clause_resolver.py) | Standalone engine evaluating claim dates against explicit per-clause `TransitionalRule`s (§5.1, §5.2, §5.3) to return the legally correct text. |
| [`retriever.py`](file:///./retriever.py) | Semantic & hybrid retriever querying local Qdrant for top-$k$ relevant clauses with exact clause boosting. |
| [`refusal_gate.py`](file:///./refusal_gate.py) | Pre-generation gate evaluating retrieval confidence, topic scope, missing date context, and contradictions before allowing generation. |
| [`answer_builder.py`](file:///./answer_builder.py) | Synthesizes answers using Gemini 2.5 Flash (with offline fallback) with strict clause-level citation traceability. |
| [`cli.py`](file:///./cli.py) | Interactive / flag-based CLI with dynamic claim date prompting and formatted terminal display. |
| [`main.py`](file:///./main.py) | End-to-end pipeline orchestrator wiring all modules. |

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
*(Note: The system automatically falls back to deterministic offline embeddings and template synthesis if `GOOGLE_API_KEY` is not present, allowing full local testing).*

---

## 📖 Usage & Execution

### 1. Ingest & Index Corpus
Rebuild the local Qdrant vector index:
```bash
python ingest.py
```

### 2. Run CLI Queries

#### Example 1: Post-Amendment Date Query (March 15, 2026)
```bash
python cli.py --query "What is the standard earnings disregard?" --date 2026-03-15
```
*Output:*
- **Status**: `GROUNDED & VERIFIED`
- **Claim Date**: `2026-03-15`
- **Cited Clauses**: `§6.4.1(a), §5.2`
- **Answer**: Cites `$260.00 per week` according to `§6.4.1(a)` as amended by Amendment 2026-01 under transitional rule `§5.2`.

#### Example 2: Pre-Amendment Date Query (February 15, 2026)
```bash
python cli.py --query "What is the standard earnings disregard?" --date 2026-02-15
```
*Output:*
- **Status**: `GROUNDED & VERIFIED`
- **Claim Date**: `2026-02-15`
- **Cited Clauses**: `§6.4.1(a)`
- **Answer**: Cites `$200.00 per week` according to base policy manual `§6.4.1(a)`.

#### Example 3: Out-of-Scope Query
```bash
python cli.py --query "What is the company policy regarding pet allowances?"
```
*Output:*
- **Status**: `REFUSED / GUARDED`
- **Reason**: `OUT_OF_SCOPE`
- **Suggested Contact**: `HR Policy Desk (hr-policy@company.internal)`

#### Example 4: Interactive Mode (Dynamic Date Prompting)
```bash
python cli.py
```

---

## 🧪 Running Automated Tests

Run the complete test suite across all modules:
```bash
pytest -v
```
