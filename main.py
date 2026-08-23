"""main.py

Orchestrator for 'The Grounded Answer' pipeline.
Wires together:
  1. Corpus Ingestion & Qdrant Vector Store
  2. Vector Retrieval
  3. Date-Aware Clause Resolution (Transitional Rules)
  4. Refusal Gate (Guardrails & Refusal Guidance)
  5. Grounded Answer Building (Strict Citations)
"""

from datetime import date
from typing import Optional, Union
from answer_builder import AnswerBuilder, GroundedAnswer
from clause_resolver import ClauseResolver
from ingest import ingest_to_qdrant, load_and_chunk_corpus
from refusal_gate import RefusalEvaluation, RefusalGate
from retriever import Retriever


class GroundedAnswerPipeline:
    """End-to-end RAG orchestrator for date-aware policy inquiries."""

    def __init__(self):
        self.resolver = ClauseResolver()
        self.refusal_gate = RefusalGate()
        self.answer_builder = AnswerBuilder()
        self.retriever: Optional[Retriever] = None

    def initialize(self, corpus_dir: str = "corpus"):
        """Ingest corpus into in-memory Qdrant and initialize retriever."""
        chunks = load_and_chunk_corpus(corpus_dir)
        client = ingest_to_qdrant(chunks)
        self.retriever = Retriever(client)

    def query(
        self,
        question: str,
        claim_date: date
    ) -> Union[GroundedAnswer, RefusalEvaluation]:
        """Execute the full RAG pipeline for a given question and claim date.

        Args:
            question: User question about policy.
            claim_date: The date on which the claim/expense occurred.

        Returns:
            GroundedAnswer if grounded and verified, or RefusalEvaluation with refusal guidance.
        """
        raise NotImplementedError("Stub: GroundedAnswerPipeline.query will be implemented in subsequent phase.")


def run_pipeline(question: str, claim_date: date) -> Union[GroundedAnswer, RefusalEvaluation]:
    """Convenience helper to initialize and run a query through the pipeline."""
    pipeline = GroundedAnswerPipeline()
    pipeline.initialize()
    return pipeline.query(question, claim_date)
