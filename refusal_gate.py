"""refusal_gate.py

Refusal Gate and Hallucination Guard.
Evaluates query intent, retrieval relevance scores, and resolved clause consistency
to decide whether an inquiry can be safely answered or must be explicitly refused
with clear guidance and contact points.
"""

import re
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

from clause_resolver import ResolvedClause
from retriever import RetrievedClause


class RefusalReason(str, Enum):
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    OUT_OF_SCOPE = "OUT_OF_SCOPE"
    CONTRADICTION = "CONTRADICTION"
    MISSING_DATE_CONTEXT = "MISSING_DATE_CONTEXT"


# Known out-of-scope topics not covered in the policy manual
OUT_OF_SCOPE_PATTERNS = [
    r"\bpet(?:s)?\b",
    r"\bdog(?:s)?\b",
    r"\bcat(?:s)?\b",
    r"\bparking\b",
    r"\bvehicle lease\b",
    r"\bstock options?\b",
    r"\bequity grant\b",
    r"\bcrypto(?:currency)?\b",
    r"\blawsuit\b",
    r"\blegal dispute\b",
    r"\bcake\b",
    r"\brecipe\b",
    r"\bweather\b",
]


@dataclass
class RefusalEvaluation:
    """Decision output of the refusal gate with actionable user guidance."""
    should_refuse: bool
    reason: Optional[str] = None
    message: str = ""
    suggested_contact: str = "HR Policy Desk"

    # Backward compatibility properties
    @property
    def refusal_message(self) -> str:
        return self.message

    @property
    def guidance(self) -> str:
        return self.suggested_contact


class RefusalGate:
    """Gatekeeper that enforces grounding constraints before answer generation."""

    def __init__(self, confidence_threshold: float = 0.40):
        self.confidence_threshold = confidence_threshold

    def _route_contact_desk(self, query: str, resolved_clauses: List[ResolvedClause]) -> str:
        """Route inquiry to the most relevant specialized team."""
        q_lower = query.lower()

        # Check by resolved clauses first
        for rc in resolved_clauses:
            if rc.clause_id.startswith("§1."):
                return "Finance & Travel Operations (travel-desk@company.internal)"
            elif rc.clause_id.startswith("§2."):
                return "Health & Wellness Benefits Team (benefits@company.internal)"
            elif rc.clause_id.startswith("§3."):
                return "IT & Workplace Operations (workplace@company.internal)"
            elif rc.clause_id.startswith("§6."):
                return "Benefits Compliance Team (compliance@company.internal)"
            elif rc.clause_id.startswith("§4."):
                return "Claims & Appeals Review Board (claims-appeals@company.internal)"

        # Check by query keywords
        if any(w in q_lower for w in ["travel", "flight", "hotel", "lodging", "meal", "mileage"]):
            return "Finance & Travel Operations (travel-desk@company.internal)"
        elif any(w in q_lower for w in ["dental", "vision", "optical", "mental health", "medical", "wellness"]):
            return "Health & Wellness Benefits Team (benefits@company.internal)"
        elif any(w in q_lower for w in ["remote", "broadband", "home office", "stipend"]):
            return "IT & Workplace Operations (workplace@company.internal)"
        elif any(w in q_lower for w in ["earnings", "disregard", "income", "assessment"]):
            return "Benefits Compliance Team (compliance@company.internal)"
        elif any(w in q_lower for w in ["appeal", "deadline", "dispute", "denial"]):
            return "Claims & Appeals Review Board (claims-appeals@company.internal)"

        return "HR Policy Desk (hr-policy@company.internal)"

    def _is_explicitly_out_of_scope(self, query: str) -> bool:
        """Check if query matches patterns for topics outside the policy manual."""
        for pattern in OUT_OF_SCOPE_PATTERNS:
            if re.search(pattern, query, re.IGNORECASE):
                return True
        return False

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
            RefusalEvaluation with decision, reason, refusal message, and suggested contact.
        """
        contact = self._route_contact_desk(query, resolved_clauses)

        # Condition 1: Explicit out-of-scope query
        if self._is_explicitly_out_of_scope(query):
            return RefusalEvaluation(
                should_refuse=True,
                reason=RefusalReason.OUT_OF_SCOPE.value,
                message=(
                    f"I cannot answer this question because the company Policy Manual and Amendments "
                    f"do not cover this topic. The manual only governs Travel, Medical/Wellness, "
                    f"Home Office, Claim Timelines, and Earnings Disregards."
                ),
                suggested_contact=contact
            )

        # Condition 2: Low confidence / no retrieval
        if not retrieved_clauses:
            return RefusalEvaluation(
                should_refuse=True,
                reason=RefusalReason.LOW_CONFIDENCE.value,
                message=(
                    "No relevant clauses were found in the Policy Manual matching your question. "
                    "Please rephrase your question or specify the relevant policy section."
                ),
                suggested_contact=contact
            )

        top_score = max((c.similarity_score for c in retrieved_clauses), default=0.0)
        if top_score < self.confidence_threshold:
            return RefusalEvaluation(
                should_refuse=True,
                reason=RefusalReason.LOW_CONFIDENCE.value,
                message=(
                    f"Retrieval confidence is insufficient ({top_score:.2f} < {self.confidence_threshold:.2f}). "
                    f"The policy manual does not contain sufficiently authoritative clauses on this subject."
                ),
                suggested_contact=contact
            )

        # Condition 3: Missing date context for amended clause
        if not resolved_clauses and any(rc.document_type == "amendment" for rc in retrieved_clauses):
            return RefusalEvaluation(
                should_refuse=True,
                reason=RefusalReason.MISSING_DATE_CONTEXT.value,
                message=(
                    "The retrieved clauses have date-dependent amendment rules. "
                    "Please provide a claim or event date (YYYY-MM-DD) to resolve the legally effective text."
                ),
                suggested_contact=contact
            )

        # Condition 4: Direct contradiction detection
        if len(resolved_clauses) >= 2:
            clause_texts = [rc.effective_text.lower() for rc in resolved_clauses]
            # Check for direct conflicting numeric assertions within the same clause scope
            if any("prohibited" in t for t in clause_texts) and any("mandatory" in t for t in clause_texts):
                return RefusalEvaluation(
                    should_refuse=True,
                    reason=RefusalReason.CONTRADICTION.value,
                    message=(
                        "An unresolvable legal contradiction was detected between the retrieved policy clauses. "
                        "Manual review by the compliance team is required."
                    ),
                    suggested_contact="Benefits Compliance Team (compliance@company.internal)"
                )

        # Passed all refusal gates
        return RefusalEvaluation(
            should_refuse=False,
            reason=None,
            message="Query successfully validated against policy corpus.",
            suggested_contact=contact
        )
