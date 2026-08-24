# AI Usage & Attribution Disclosure

## 1. Overview & Human Oversight
This project was built leveraging AI coding assistants (**Antigravity Agent by Google DeepMind** and **Claude**) for rapid prototyping, module scaffolding, and iterative refactoring. All architecture decisions, domain taxonomy definitions, safety threshold policies, and final code verifications were human-guided and thoroughly validated through automated testing (`pytest`).

---

## 2. Iterative Development & AI Assistance Workflow

* **Phase 1: Architecture & System Design**
  * **Human Role:** Defined core system objectives, temporal policy cutover requirements, vector DB selection (Qdrant), and strict citation rules.
  * **AI Assistance:** Helped map system workflow into modular Python packages (`ingest`, `retriever`, `clause_resolver`, `refusal_gate`, `answer_builder`).

* **Phase 2: Ingestion & Boundary Parsing (`ingest.py`)**
  * **Human Role:** Defined policy manual parsing rules (`§P.S.p` clause boundaries) and transitional metadata structures.
  * **AI Assistance:** Accelerated regex pattern matching for section header parsing and amendment metadata extraction.

* **Phase 3: Date-Aware & Transitional Resolution (`clause_resolver.py`)**
  * **Human Role:** Specified transitional legal logic (§5.1 Standard, §5.2 Event-based, §5.3 Period-spanning).
  * **AI Assistance:** Drafted evaluation logic for date comparison against policy effective cutovers.

* **Phase 4: Vector Retrieval & Domain Filtering (`retriever.py`)**
  * **Human Role:** Identified cross-domain leakage issue (e.g., optical benefits returning on earnings disregard queries); established minimum similarity threshold (`0.65`) and domain taxonomy filtering.
  * **AI Assistance:** Implemented Qdrant query scoring, metadata filtering, and domain re-ranking logic.

* **Phase 5: Refusal Gate & Safety Guardrails (`refusal_gate.py`)**
  * **Human Role:** Designed fallback routing rules and HR contact delegation for out-of-scope or ungrounded queries.
  * **AI Assistance:** Generated guardrail validation checks and structured fallback responses.

* **Phase 6: Grounded Synthesis Engine (`answer_builder.py`)**
  * **Human Role:** Authored strict grounding system prompts mandating inline section citations (`§6.4.1(a)`) and active claim date context.
  * **AI Assistance:** Formatted LangChain pipeline templates and Gemini 2.5 Flash invocation blocks.

* **Phase 7: CLI Interface & Testing Framework (`cli.py`, `tests/`)**
  * **Human Role:** Designed user-facing CLI behavior, interactive date prompting, and test suite requirements.
  * **AI Assistance:** Generated unit test cases in `pytest` to achieve 30 passing assertions covering core e2e workflows.

---

## 3. Prompts & Verification Summary
* **Code Verification:** 100% of AI-generated code was manually reviewed, executed locally in a Python 3.11 virtual environment, and validated against standard unit/integration test suites.
* **Refinement Iterations:** System prompts and vector thresholds were iteratively tuned to ensure zero hallucinations and complete prevention of out-of-domain context leakage.