"""refusal_gate.py

Evaluates whether a grounded answer can be confidently generated.
Produces an explicit refusal with guidance if:
  (a) Retrieval confidence is below threshold
  (b) The policy manual does not cover the topic
  (c) Retrieved clauses contradict or have unresolvable conflicts
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional
from clause_resolver import ResolvedClause
from retriever import RetrievedClause


class RefusalReason(str, Enum):
    LOW_CONFIDENCE = "low_retrieval_confidence"
    OUT_OF_SCOPE = "topic_not_covered_in_manual"
    CONTRADICTORY_CLAUSES = "unresolvable_contradiction"
    INSUFFICIENT_INFORMATION = "insufficient_information"


@dataclass
class RefusalEvaluation:
    """Decision output of the refusal gate."""
    should_refuse: bool
    reason: Optional[RefusalReason] = None
    refusal_message: Optional[str] = None
    guidance: Optional[str] = None


class RefusalGate:
    """Gatekeeper that enforces grounding constraints before answer generation."""

    def __init__(self, confidence_threshold: float = 0.65):
        self.confidence_threshold = confidence_threshold

    def evaluate(
        self,
        query: str,
        retrieved_clauses: List[RetrievedClause],
        resolved_clauses: List[ResolvedClause]
    ) -> RefusalEvaluation:
        """Analyze query and candidate clauses to decide whether to proceed or refuse.

        Args:
            query: The user question.
            retrieved_clauses: Raw vector retrieval results with scores.
            resolved_clauses: Date-resolved clause instances.

        Returns:
            RefusalEvaluation containing should_refuse boolean and refusal explanation.
        """
        raise NotImplementedError("Stub: RefusalGate.evaluate will be implemented in subsequent phase.")
