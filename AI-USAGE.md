# AI Usage & Attribution

## Hackathon Architecture & Scaffolding
- **Antigravity Agent (Google DeepMind)** & **Claude**: Used for end-to-end project scaffolding, modular interface design, and iterative implementation.
- **Iterative Implementation Workflow**:
  - **Phase 1**: Core skeleton, module contracts, requirements, and environment setup.
  - **Phase 2**: Corpus parsing, `§P.S.p` boundary chunking, and amendment-to-transitional rule linkage (`ingest.py`).
  - **Phase 3**: Date-aware legal resolution and structured transitional rule evaluation (`clause_resolver.py`).
  - **Phase 4**: Local Qdrant vector retrieval, clause boosting, and deduplication (`retriever.py`).
  - **Phase 5**: Guardrail refusal gate for low confidence, out-of-scope topics, and contradictions (`refusal_gate.py`).
  - **Phase 6**: Grounded synthesis with Gemini 2.5 Flash and clause citations (`answer_builder.py`).
  - **Phase 7**: Orchestration (`main.py`), interactive CLI with date prompting (`cli.py`), and end-to-end test integration (`tests/test_pipeline_e2e.py`).
