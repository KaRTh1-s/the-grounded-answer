"""clause_resolver.py

Given a clause number (e.g. §1.1.1) and a claim date, resolves the legally
correct text (base manual vs. amended provision) according to explicit
transitional applicability rules modeled as structured data.
"""

from dataclasses import dataclass
from datetime import date
from enum import Enum
from typing import Callable, Dict, List, Optional


class ClauseVersion(str, Enum):
    BASE = "base"
    AMENDED = "amended"


@dataclass(frozen=True)
class TransitionalRule:
    """Structured representation of an amendment transitional rule."""
    rule_id: str
    description: str
    effective_threshold: date
    # Criterion condition to determine if amended text applies for a given claim date
    applicability_condition: Callable[[date], bool]


@dataclass
class ResolvedClause:
    """The result of resolving a clause for a specific claim date."""
    clause_id: str
    version_applied: ClauseVersion
    resolved_text: str
    transitional_rule_applied: Optional[str]
    claim_date: date
    source_reference: str


class ClauseResolver:
    """Standalone, testable resolver that maps clause IDs and claim dates to legally effective text."""

    def __init__(self, rules: Optional[Dict[str, TransitionalRule]] = None):
        self.rules: Dict[str, TransitionalRule] = rules or {}

    def resolve(self, clause_id: str, claim_date: date) -> ResolvedClause:
        """Resolve the legally binding text for a specific clause as of the given claim date.

        Args:
            clause_id: Clause identifier (e.g. '§1.1.1' or '§4.3.1').
            claim_date: The date on which the claim or expense occurred.

        Returns:
            ResolvedClause detailing text, version used, and applied transitional rule.
        """
        raise NotImplementedError("Stub: ClauseResolver.resolve will be implemented in subsequent phase.")

    def resolve_batch(self, clause_ids: List[str], claim_date: date) -> List[ResolvedClause]:
        """Resolve multiple clauses for a single claim date."""
        return [self.resolve(cid, claim_date) for cid in clause_ids]
