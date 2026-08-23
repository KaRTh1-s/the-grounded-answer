# The Grounded Answer

> **A Date-Aware CLI Policy RAG Assistant with Clause-Level Citation Grounding & Guardrails**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Tests Passing](https://img.shields.io/badge/pytest-30%20passed-brightgreen.svg)](tests/)
[![Vector Store](https://img.shields.io/badge/Qdrant-in--memory-red.svg)](https://qdrant.tech/)
[![LLM Orchestration](https://img.shields.io/badge/LangChain-Google%20GenAI-orange.svg)](https://www.langchain.com/)

**The Grounded Answer** is an authoritative, date-aware CLI RAG assistant built for policy inquiries. It parses policy manuals down to exact clause boundaries (`§P.S.p`), models amendment cutover rules as structured data evaluations (Part 5 transitional rules `§5.1`, `§5.2`, `§5.3`), retrieves clauses with zero external hosting dependencies via local in-memory Qdrant, and enforces a strict pre-generation **Refusal Gate** to eliminate hallucinations.

---

## 🌟 Core Features

- **Clause-Level Granularity (`§P.S.p`)**: Chunks documents strictly along natural clause boundaries (`§1.1.1`, `§6.4.1(a)`), preserving hierarchical metadata (Parts, Sections, and Titles).
- **Date-Aware Transitional Resolution Engine**: Evaluates claim and event dates against structured transitional provisions rather than vague prompt heuristics:
  - **Rule §5.1 (Standard Cutover Date)**: Automatically applies amended provisions for claims on or after `2026-03-01`; retains base manual text for earlier dates.
  - **Rule §5.2 (Event / Assessment Period Date)**: Evaluates assessment periods or changes of circumstance against the March 1, 2026 threshold (e.g. for earnings disregard `§6.4.1(a)`).
  - **Rule §5.3 (Period-Spanning Claims)**: Detects claims spanning across the cutover date and computes pro-rata dual-period apportionment.
- **Hybrid Retrieval & Exact Matching**: Blends semantic embeddings with keyword scoring and exact clause identifier boosting (`1.0` confidence).
- **Refusal Gate & Hallucination Guard**: Intercepts queries before LLM generation and produces explicit refusals with domain-specific contact routing if:
  - Retrieval confidence is low ($< 0.40$).
  - The topic is uncovered/out-of-scope (e.g. pets, parking, equity grants).
  - Date context is missing for an amended provision.
  - Direct legal contradictions exist between active clauses.
- **Strict Citation Grounding**: Synthesizes verified answers citing exact clause numbers and governing transitional rules using Google Gemini (`gemini-2.5-flash`), with deterministic offline fallback for offline test resilience.

---

## 🏗️ Architecture & Module Separability

The system is built as clean, standalone, testable components with zero circular dependencies:

```
                      +-----------------------------+
                      |         Corpus Docs         |
                      | (policy-manual & amendment) |
                      +--------------+--------------+
                                     |
                                     v
                           +-------------------+
                           |     ingest.py     |
                           | (Chunk & Vector)  |
                           +---------+---------+
                                     |
                                     v
                           +-------------------+
                           |  Qdrant In-Memory |
                           |  (policy_corpus)  |
                           +---------+---------+
                                     |
                                     v
 [User Query] +--------->  +-------------------+
 [Claim Date]              |    retriever.py   |  <--- Top-k Hybrid Search &
                           | (PolicyRetriever) |       Exact Clause Boosting
                           +---------+---------+
                                     |
                                     v
                           +-------------------+
                           | clause_resolver.py|  <--- Evaluates Part 5 Rules
                           |  (ClauseResolver) |       (§5.1, §5.2, §5.3)
                           +---------+---------+
                                     |
                                     v
                           +-------------------+
                           |  refusal_gate.py  |  <--- Hallucination Guard:
                           |   (RefusalGate)   |       Low Confidence / Out of Scope
                           +---------+---------+
                                     |
                       +-------------+-------------+
                       |                           |
                [If Refused]                  [If Passed]
                       |                           |
                       v                           v
             +--------------------+      +--------------------+
             | RefusalEvaluation  |      |  answer_builder.py |
             | (Contact Routing)  |      |   (AnswerBuilder)  |
             +--------------------+      +---------+----------+
                                                   |
                                                   v
                                         +--------------------+
                                         |   GroundedAnswer   |
                                         |  (Clause Citation) |
                                         +--------------------+
                                                   |
                                                   v
                                         +--------------------+
                                         |     main.py /      |
                                         |       cli.py       |
                                         +--------------------+
```

### Module Responsibilities

| Module | File | Responsibility |
|---|---|---|
| **Corpus Ingestion** | [`ingest.py`](file:///./ingest.py) | Parses markdown by `§P.S.p` clause boundaries, links amendment items to Part 5 transitional rules, and indexes points into local Qdrant. |
| **Date Resolution Engine** | [`clause_resolver.py`](file:///./clause_resolver.py) | Standalone engine evaluating claim dates against explicit per-clause `TransitionalRule`s (§5.1, §5.2, §5.3) to return legally effective text. |
| **Vector Retriever** | [`retriever.py`](file:///./retriever.py) | Queries local Qdrant collection for top-$k$ relevant clauses with exact clause ID boosting and deduplication. |
| **Refusal Gate** | [`refusal_gate.py`](file:///./refusal_gate.py) | Enforces pre-generation guardrails for low confidence, out-of-scope topics, missing date context, and contradictions. |
| **Answer Builder** | [`answer_builder.py`](file:///./answer_builder.py) | Synthesizes grounded answers using Gemini 2.5 Flash with strict clause-level citation traceability and offline fallback. |
| **CLI Entrypoint** | [`cli.py`](file:///./cli.py) | Command-line interface with argument parsing, interactive date prompting, and clean status reporting. |
| **Pipeline Orchestrator**| [`main.py`](file:///./main.py) | End-to-end orchestrator wiring all modules. |

---

## 🚀 Mechanical Installation & Execution Guide

### 1. Prerequisites
- **Python 3.11+** installed
- **Git** installed

### 2. Virtual Environment Setup
```bash
# Clone repository and navigate into project root
git clone <REPO_URL>
cd the-grounded-answer

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# Windows (Command Prompt):
.\venv\Scripts\activate.bat
# Linux / macOS:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Copy `.env.example` to `.env` and provide your Google Gemini API key:
```bash
cp .env.example .env
```
Inside `.env`:
```env
GOOGLE_API_KEY=your_google_api_key_here
```
*(Note: If `GOOGLE_API_KEY` is omitted, the system automatically falls back to deterministic offline embeddings and template synthesis for complete offline execution and testing).*

### 5. Ingest Corpus into Qdrant
Rebuild the local in-memory vector index:
```bash
python ingest.py
```
*Expected Output:*
```text
============================================================
  The Grounded Answer - Corpus Ingestion Engine
============================================================

[+] Total Base Clauses Parsed: 16
[+] Total Amendments Parsed:   6
[+] Total Chunks Indexed:      22
[OK] Successfully indexed 22 vectors in Qdrant collection 'policy_corpus'.
```

---

## 💻 CLI Usage & Examples

### Example 1: Pre-Cutover Claim Date (`2026-02-15`)
Evaluates against base policy manual rate ($200.00):
```bash
python cli.py --query "What is the standard weekly earnings disregard?" --date 2026-02-15
```
*Output:*
```text
=================================================================
      THE GROUNDED ANSWER - DATE-AWARE POLICY ASSISTANT
=================================================================

[QUESTION]: What is the standard weekly earnings disregard?
-----------------------------------------------------------------

[STATUS]: GROUNDED & VERIFIED
[CLAIM DATE]: 2026-02-15
[CITED CLAUSES]: §6.4.1(a), §6.4.1(b), §6.4.2

[ANSWER]:
For claim date 2026-02-15:

According to §6.4.1(a) of the base policy manual, §6.4.1(a) The standard weekly earnings disregard for single claimants without dependants is $200.00 per week..
=================================================================
```

---

### Example 2: Post-Cutover Claim Date (`2026-03-15`)
Evaluates against amended provision ($260.00) under transitional rule `§5.2`:
```bash
python cli.py --query "What is the standard weekly earnings disregard?" --date 2026-03-15
```
*Output:*
```text
=================================================================
      THE GROUNDED ANSWER - DATE-AWARE POLICY ASSISTANT
=================================================================

[QUESTION]: What is the standard weekly earnings disregard?
-----------------------------------------------------------------

[STATUS]: GROUNDED & VERIFIED
[CLAIM DATE]: 2026-03-15
[CITED CLAUSES]: §5.2, §6.4.1(a), §6.4.1(b)

[TRANSITIONAL RATIONALE]:
§6.4.1(a): §5.2 Event-based / Assessment Period Effective Date - Event/claim date (2026-03-15) is on or after 1 March 2026. Transitional rule §5.2 applies the amended rate from amendment-2026-01.md.

[ANSWER]:
For claim date 2026-03-15:

According to §6.4.1(a) (as amended by amendment-2026-01), **2.1** In §6.4.1(a), for "$200.00 per week" substitute "$260.00 per week"..

Applicable transitional provision: §5.2 Event-based / Assessment Period Effective Date (Event/claim date (2026-03-15) is on or after 1 March 2026. Transitional rule §5.2 applies the amended rate from amendment-2026-01.md.).
=================================================================
```

---

### Example 3: Out-of-Scope Query Refusal
Demonstrates guardrail interception with contact routing:
```bash
python cli.py --query "What is the policy for bringing pets into the office?"
```
*Output:*
```text
=================================================================
      THE GROUNDED ANSWER - DATE-AWARE POLICY ASSISTANT
=================================================================

[QUESTION]: What is the policy for bringing pets into the office?
-----------------------------------------------------------------

[STATUS]: REFUSED / GUARDED
[REASON]: OUT_OF_SCOPE

[EXPLANATION]:
I cannot answer this question because the company Policy Manual and Amendments do not cover this topic. The manual only governs Travel, Medical/Wellness, Home Office, Claim Timelines, and Earnings Disregards.

[SUGGESTED CONTACT]: HR Policy Desk (hr-policy@company.internal)
=================================================================
```

---

### Example 4: Interactive Mode with Dynamic Date Prompting
If an amended clause is queried without a date flag, the CLI automatically prompts:
```bash
python cli.py
```
```text
Enter your policy question: What is the lodging reimbursement limit?

[!] The retrieved policy provisions contain date-dependent amendment rules.
Please enter the claim date (YYYY-MM-DD): 2026-03-15
```

---

## 🧪 Running Automated Tests

Run the complete test suite with verbose reporting:
```bash
pytest -v
```

*Output:*
```text
tests/test_answer_builder.py::test_build_answer_with_clause_citation PASSED [  3%]
tests/test_answer_builder.py::test_build_answer_base_unamended_clause PASSED [  6%]
tests/test_answer_builder.py::test_build_answer_empty_resolved_clauses PASSED [ 10%]
tests/test_clause_resolver.py::test_unamended_clause_always_returns_base_text PASSED [ 13%]
tests/test_clause_resolver.py::test_earnings_disregard_pre_march_returns_base PASSED [ 16%]
tests/test_clause_resolver.py::test_earnings_disregard_post_march_returns_amended PASSED [ 20%]
tests/test_clause_resolver.py::test_rule_5_1_standard_travel_allowance PASSED [ 23%]
tests/test_clause_resolver.py::test_period_spanning_claims_rule_5_3 PASSED [ 26%]
tests/test_clause_resolver.py::test_missing_date_for_amended_clause_raises_error PASSED [ 30%]
tests/test_clause_resolver.py::test_invalid_clause_id_raises_keyerror PASSED [ 33%]
tests/test_ingest.py::test_base_manual_parses_clause_6_4_1_a PASSED      [ 36%]
tests/test_ingest.py::test_amendment_earnings_disregard_links_to_transitional_rule PASSED [ 40%]
tests/test_ingest.py::test_total_chunk_count_and_accuracy PASSED         [ 43%]
tests/test_ingest.py::test_ingest_to_qdrant_runs_cleanly PASSED          [ 46%]
tests/test_pipeline_e2e.py::test_e2e_pre_march_earnings_disregard PASSED [ 50%]
tests/test_pipeline_e2e.py::test_e2e_post_march_earnings_disregard PASSED [ 53%]
tests/test_pipeline_e2e.py::test_e2e_travel_lodging_amendment PASSED     [ 56%]
tests/test_pipeline_e2e.py::test_e2e_out_of_scope_refusal PASSED         [ 60%]
tests/test_pipeline_e2e.py::test_e2e_missing_date_on_amended_clause_refusal PASSED [ 63%]
tests/test_refusal_gate.py::test_out_of_scope_query_refusal PASSED       [ 66%]
tests/test_refusal_gate.py::test_low_confidence_retrieval_refusal PASSED [ 70%]
tests/test_refusal_gate.py::test_valid_policy_query_passes PASSED        [ 73%]
tests/test_refusal_gate.py::test_contact_routing_for_travel_domain PASSED [ 76%]
tests/test_refusal_gate.py::test_contradiction_detection PASSED          [ 80%]
tests/test_retriever.py::test_retrieve_earnings_disregard PASSED         [ 83%]
tests/test_retriever.py::test_retrieve_outpatient_medical_claims PASSED  [ 86%]
tests/test_retriever.py::test_exact_clause_id_query_prioritization PASSED [ 90%]
tests/test_retriever.py::test_low_relevance_query_filtering PASSED       [ 93%]
tests/test_retriever.py::test_retrieved_clause_structure PASSED          [ 96%]
tests/test_scaffold.py::test_module_interfaces_and_types PASSED          [100%]

============================= 30 passed in 1.35s ==============================
```

---

## 📁 Repository Deliverables Checklist

- [x] **[`requirements.txt`](file:///./requirements.txt)**: Core dependencies (`langchain`, `qdrant-client`, `google-generativeai`, `pydantic`, `pytest`).
- [x] **[`.env.example`](file:///./.env.example)**: Environment template with `.env` git-ignored.
- [x] **[`README.md`](file:///./README.md)**: Full mechanical clone, setup, ingestion, execution, and test guide.
- [x] **[`DECISIONS.md`](file:///./DECISIONS.md)**: Architecture rationale logs covering technology stack, chunking strategy, structured transitional rules, hybrid scoring, and refusal gates.
- [x] **[`AI-USAGE.md`](file:///./AI-USAGE.md)**: AI attribution and prompt-based pair programming workflow notes.
- [x] **[`corpus/`](file:///./corpus/)**: Structured policy manual (`policy-manual.md`) and amendment document (`amendment-2026-01.md`).
- [x] **[`tests/`](file:///./tests/)**: 30 comprehensive unit and end-to-end integration tests.
