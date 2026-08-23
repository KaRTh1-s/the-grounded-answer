"""answer_builder.py

Given retrieved clauses and date-resolved text, constructs a strictly grounded
answer using Gemini 2.5 Flash where every claim explicitly cites a clause number (§x.x.x).
"""

from dataclasses import dataclass, field
from datetime import date
from typing import List, Optional
from clause_resolver import ResolvedClause


@dataclass
class Citation:
    """Explicit citation linking an answer claim to a specific clause and rule."""
    clause_id: str
    version: str
    transitional_rule: Optional[str]
    source_quote: str


@dataclass
class GroundedAnswer:
    """Complete grounded answer with citation provenance."""
    question: str
    claim_date: date
    answer_text: str
    citations: List[Citation] = field(default_factory=list)
    confidence: float = 1.0


class AnswerBuilder:
    """Constructs verifiable, grounded answers from date-resolved clauses."""

    def __init__(self, model_name: str = "gemini-2.5-flash"):
        self.model_name = model_name

    def build_answer(
        self,
        query: str,
        claim_date: date,
        resolved_clauses: List[ResolvedClause]
    ) -> GroundedAnswer:
        """Synthesize answer with citations from the provided resolved clauses.

        Args:
            query: User's question.
            claim_date: Effective date of the claim.
            resolved_clauses: Date-resolved legally correct clause texts.

        Returns:
            GroundedAnswer containing answer text and citation list.
        """
        raise NotImplementedError("Stub: AnswerBuilder.build_answer will be implemented in subsequent phase.")
