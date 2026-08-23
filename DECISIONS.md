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
*Future decision logs will be appended here as implementation proceeds.*
