"""main.py

End-to-End Orchestration Engine for 'The Grounded Answer'.
Wires together:
  1. Semantic Vector Retrieval (PolicyRetriever)
  2. Date-Aware Clause Resolution (ClauseResolver)
  3. Grounding & Refusal Gate (RefusalGate)
  4. Grounded Answer Synthesis (AnswerBuilder)
"""

from datetime import date, datetime
from typing import List, Optional, Union

from answer_builder import AnswerBuilder, GroundedAnswer
from clause_resolver import ClauseResolver, DateRequiredError, ResolvedClause
from refusal_gate import RefusalEvaluation, RefusalGate, RefusalReason
from retriever import PolicyRetriever, RetrievedClause


class GroundedAnswerPipeline:
    """Orchestrates end-to-end date-aware retrieval-augmented generation."""

    def __init__(
        self,
        retriever: Optional[PolicyRetriever] = None,
        resolver: Optional[ClauseResolver] = None,
        refusal_gate: Optional[RefusalGate] = None,
        answer_builder: Optional[AnswerBuilder] = None
    ):
        self.resolver = resolver or ClauseResolver()
        self.retriever = retriever or PolicyRetriever()
        self.refusal_gate = refusal_gate or RefusalGate()
        self.answer_builder = answer_builder or AnswerBuilder()

    def _parse_date(self, d: Optional[Union[str, date]]) -> Optional[date]:
        """Convert string date input to datetime.date object."""
        if d is None or isinstance(d, date):
            return d
        if isinstance(d, str):
            try:
                return datetime.strptime(d.strip(), "%Y-%m-%d").date()
            except ValueError:
                return None
        return None

    def requires_date_context(self, query: str) -> bool:
        """Check if retrieved candidate clauses for a query require a date context."""
        if self.refusal_gate._is_explicitly_out_of_scope(query):
            return False

        candidates = self.retriever.retrieve(query, top_k=3, score_threshold=0.35)
        for cand in candidates:
            if cand.document_type == "amendment" or self.resolver.is_clause_amended(cand.clause_id):
                return True
        return False

    def run_query(
        self,
        query: str,
        claim_date: Optional[Union[str, date]] = None,
        event_date: Optional[Union[str, date]] = None
    ) -> Union[GroundedAnswer, RefusalEvaluation]:
        """Execute the full RAG pipeline for a given query and date context.

        Args:
            query: User question regarding company policy.
            claim_date: Effective date on which the claim/expense was incurred.
            event_date: Optional specific change of circumstances date.

        Returns:
            GroundedAnswer if verifiable, or RefusalEvaluation if refused.
        """
        # Step 0: Fast pre-check for explicitly out of scope queries
        if self.refusal_gate._is_explicitly_out_of_scope(query):
            return self.refusal_gate.evaluate(query=query, retrieved_clauses=[], resolved_clauses=[])

        parsed_claim_date = self._parse_date(claim_date)
        parsed_event_date = self._parse_date(event_date)

        # Step 1: Semantic & Exact Retrieval
        retrieved_clauses: List[RetrievedClause] = self.retriever.retrieve(query, top_k=4)

        # Step 2: Date-Aware Clause Resolution
        resolved_clauses: List[ResolvedClause] = []
        for rc in retrieved_clauses:
            try:
                resolved = self.resolver.resolve_clause(
                    clause_id=rc.clause_id,
                    claim_date=parsed_claim_date,
                    event_date=parsed_event_date
                )
                resolved_clauses.append(resolved)
            except DateRequiredError:
                # If date was omitted on an amended clause, forward to RefusalGate
                return RefusalEvaluation(
                    should_refuse=True,
                    reason=RefusalReason.MISSING_DATE_CONTEXT.value,
                    message=(
                        f"Clause {rc.clause_id} is governed by a date-sensitive transitional rule. "
                        f"Please provide a claim date (YYYY-MM-DD) to resolve the legally correct rate."
                    ),
                    suggested_contact="HR Policy Desk"
                )
            except KeyError:
                continue

        # Step 3: Refusal Gate Evaluation
        eval_result = self.refusal_gate.evaluate(
            query=query,
            retrieved_clauses=retrieved_clauses,
            resolved_clauses=resolved_clauses
        )

        if eval_result.should_refuse:
            return eval_result

        # Step 4: Grounded Answer Synthesis
        answer = self.answer_builder.build_answer(
            query=query,
            resolved_clauses=resolved_clauses,
            claim_date=parsed_claim_date
        )

        return answer

    # Alias for pipeline execution
    def query(
        self,
        question: str,
        claim_date: Optional[Union[str, date]] = None
    ) -> Union[GroundedAnswer, RefusalEvaluation]:
        return self.run_query(query=question, claim_date=claim_date)


def run_pipeline(
    question: str,
    claim_date: Optional[Union[str, date]] = None
) -> Union[GroundedAnswer, RefusalEvaluation]:
    """Convenience functional interface to execute pipeline queries."""
    pipeline = GroundedAnswerPipeline()
    return pipeline.run_query(query=question, claim_date=claim_date)
