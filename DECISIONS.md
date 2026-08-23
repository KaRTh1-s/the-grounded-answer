# Architecture & Technical Decisions

## 1. Why This Stack
- **LangChain for Orchestration**:
  - Provides modular abstractions for prompt templating, document loaders, and output parsing.
  - Keeps module interfaces clean and easily mockable for unit testing across pipeline stages.

- **Qdrant (Local / In-Memory)**:
  - Running local in-memory via `qdrant-client` (`location=":memory:"`) eliminates external infrastructure dependencies and hosting friction.
  - Supports rich payload metadata filtering (e.g. clause numbers, parts, sections) and deterministic state teardown in automated tests.

- **Google Gemini API (`gemini-2.5-flash` + Google Embeddings)**:
  - High inference speed and cost-effective generation.
  - Large context window and strong instruction following for strict clause citation grounding and zero-hallucination refusal behavior.

---

## 2. Ingestion & Clause Chunking Strategy (`ingest.py`)
- **Clause-Level Granularity**: Rather than fixed token-size chunking, the corpus is split along natural `§P.S.p` clause boundaries. This ensures every indexed document corresponds to an exact legal clause.
- **Explicit Transitional Linking**: Amendment provisions in Part 1–4 are parsed and explicitly linked to their governing Part 5 transitional rules (`§5.1`, `§5.2`, `§5.3`) at ingestion time rather than leaving rule resolution to ambiguous prompt heuristics.

---

## 3. Date-Aware Transitional Rule Modeling (`clause_resolver.py`)
- **Rules as Structured Data**: Rather than burying `if-else` date logic inside prompts or LLM calls, transitional rules are defined as explicit structured data models with precise threshold dates (`2026-03-01`) and condition semantics:
  - `§5.1`: Standard Cutover / Determination Date.
  - `§5.2`: Event / Assessment Period Date.
  - `§5.3`: Period-Spanning Pro-Rata Apportionment.
- **Fail-Fast Date Requirements**: When resolving an amended clause without a provided date, a `DateRequiredError` is raised to ensure the pipeline or CLI prompts the user rather than guessing.

---

## 4. Hybrid Retrieval & Clause Prioritization (`retriever.py`)
- **Clause Pattern Boosting**: Queries explicitly naming a clause (e.g. *"What does clause §4.3.1 say?"*) trigger exact clause pattern extraction and score boosting (similarity score `1.0`).
- **Hybrid Vector + Keyword Ranking**: Blends semantic vector cosine distance with keyword overlap across titles and text to maximize precision across short and specific legal queries.

---

## 5. Refusal Gate & Hallucination Guard (`refusal_gate.py`)
- **Pre-Generation Safety Boundary**: The refusal gate sits strictly between retrieval/resolution and LLM generation. If confidence is low, the topic is out-of-scope, date context is missing, or active clauses contradict, execution halts immediately with a structured refusal and domain-specific contact routing.

---

## 6. Strict Grounding & Offline Resilience (`answer_builder.py` & `main.py`)
- **Strict Provenance**: Generation is constrained exclusively to the resolved legally effective text. Every entitlement claim cites its clause ID and governing transitional provision.
- **Graceful Offline Fallback**: If an API key is not configured or network access is unavailable, deterministic template synthesis guarantees uninterrupted execution and reliable unit testing.
