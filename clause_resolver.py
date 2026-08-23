"""clause_resolver.py

Date-aware clause resolution and transitional rule evaluation engine.
Resolves the legally correct clause text (base manual vs. amendment) for a
given claim date, event date, or date period, modeling transitional rules
as explicit structured data.
"""

from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from ingest import ClauseChunk, load_and_chunk_corpus


class ClauseVersion(str, Enum):
    BASE = "base"
    AMENDED = "amendment-2026-01"


class DateRequiredError(ValueError):
    """Raised when a claim/event date is required to resolve an amended clause but was omitted."""
    def __init__(self, clause_id: str, rule_id: Optional[str] = None, message: Optional[str] = None):
        self.clause_id = clause_id
        self.rule_id = rule_id
        msg = message or f"Date is required to resolve amended clause '{clause_id}' under transitional rule {rule_id or 'applicable'}."
        super().__init__(msg)


@dataclass(frozen=True)
class TransitionalRule:
    """Structured representation of a transitional rule and its evaluation criteria."""
    rule_id: str  # e.g., "§5.1", "§5.2", "§5.3"
    description: str
    effective_threshold: date
    evaluation_logic: str = ""
    name: str = ""
    applicability_condition: Optional[Callable[[date], bool]] = None


TransitionalRuleDefinition = TransitionalRule


@dataclass
class ResolvedClause:
    """The legally binding resolution of a clause for a given date context."""
    clause_id: str
    effective_text: str = ""
    applied_source: str = "base"  # "base" | "amendment-2026-01"
    transitional_rule_applied: str = ""  # e.g., "§5.1 Standard Effective Date"
    is_amended: bool = False
    explanation: str = ""
    claim_date: Optional[date] = None
    event_date: Optional[date] = None

    # Backward compatibility fields / init
    version_applied: Optional[str] = None
    resolved_text: Optional[str] = None
    source_reference: Optional[str] = None

    def __post_init__(self):
        if not self.effective_text and self.resolved_text:
            self.effective_text = self.resolved_text
        if not self.resolved_text and self.effective_text:
            self.resolved_text = self.effective_text

        if self.version_applied and self.applied_source == "base":
            self.applied_source = str(self.version_applied)
        if not self.version_applied:
            self.version_applied = self.applied_source

        if self.source_reference and not self.explanation:
            self.explanation = f"Resolved from {self.source_reference}"
        if not self.source_reference:
            self.source_reference = self.applied_source


# Structured registry of transitional rules
TRANSITIONAL_RULES: Dict[str, TransitionalRuleDefinition] = {
    "§5.1": TransitionalRuleDefinition(
        rule_id="§5.1",
        name="§5.1 Standard Effective Date",
        description="Applies to claims or expenses incurred on or after 1 March 2026. Claims incurred prior to 1 March 2026 remain governed by base policy manual.",
        effective_threshold=date(2026, 3, 1),
        evaluation_logic="claim_date >= 2026-03-01"
    ),
    "§5.2": TransitionalRuleDefinition(
        rule_id="§5.2",
        name="§5.2 Event-based / Assessment Period Effective Date",
        description="Applies to assessment periods or changes of circumstances occurring on or after 1 March 2026. Prior claims continue at base rate until the conclusion of that assessment period.",
        effective_threshold=date(2026, 3, 1),
        evaluation_logic="event_date >= 2026-03-01 or (claim_date >= 2026-03-01 without prior assessment)"
    ),
    "§5.3": TransitionalRuleDefinition(
        rule_id="§5.3",
        name="§5.3 Period-Spanning Apportionment Rule",
        description="Where a claim or entitlement spans before and after 1 March 2026, pro-rata apportionment applies across respective date segments.",
        effective_threshold=date(2026, 3, 1),
        evaluation_logic="start_date < 2026-03-01 <= end_date"
    ),
}


class ClauseResolver:
    """Standalone, testable resolver mapping clause IDs and dates to legally effective text."""

    def __init__(
        self,
        corpus_dir: str | Path = "corpus",
        chunks: Optional[List[ClauseChunk]] = None
    ):
        self.corpus_dir = Path(corpus_dir)
        self.rules = TRANSITIONAL_RULES

        if chunks is None and self.corpus_dir.exists():
            chunks = load_and_chunk_corpus(self.corpus_dir)
        self.chunks: List[ClauseChunk] = chunks or []

        # Index base clauses and amendments
        self.base_clauses: Dict[str, ClauseChunk] = {}
        self.amendments_by_base: Dict[str, ClauseChunk] = {}

        for chunk in self.chunks:
            if chunk.document_type == "base":
                self.base_clauses[chunk.clause_id] = chunk
            elif chunk.document_type == "amendment" and chunk.amended_clause_id:
                self.amendments_by_base[chunk.amended_clause_id] = chunk

    def is_clause_amended(self, clause_id: str) -> bool:
        """Check if a given clause has any associated amendment in the corpus."""
        return clause_id in self.amendments_by_base

    def resolve_clause(
        self,
        clause_id: str,
        claim_date: Optional[date] = None,
        event_date: Optional[date] = None,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> ResolvedClause:
        """Resolve the legally correct text for a clause given date parameters.

        Args:
            clause_id: Identifier of clause (e.g. '§6.4.1(a)', '§1.1.1', '§1.1.2')
            claim_date: Date the expense or claim was incurred/submitted.
            event_date: Date of specific circumstance change or assessment period start.
            start_date: Optional start date for period-spanning claims.
            end_date: Optional end date for period-spanning claims.

        Returns:
            ResolvedClause with effective text, applied source, rule, and explanation.

        Raises:
            KeyError: If clause_id is not found in the base policy manual.
            DateRequiredError: If clause is amended and required date is missing.
        """
        # Clean clause ID formatting
        cid = clause_id.strip()
        if not cid.startswith("§") and not cid.startswith("AMEND-"):
            cid = f"§{cid}"

        # Check if clause exists in base manual
        base_chunk = self.base_clauses.get(cid)
        if not base_chunk:
            # Check if it was directly queried by amendment ID or exists
            raise KeyError(f"Clause '{cid}' not found in policy manual.")

        # Check if an amendment exists for this clause
        amend_chunk = self.amendments_by_base.get(cid)

        # Case 1: Unamended Clause
        if not amend_chunk:
            return ResolvedClause(
                clause_id=cid,
                effective_text=base_chunk.content,
                applied_source="base",
                transitional_rule_applied="None (Unamended)",
                is_amended=False,
                explanation=f"Clause {cid} has not been modified by any amendment. The base manual text applies universally.",
                claim_date=claim_date,
                event_date=event_date,
            )

        # Case 2: Clause is amended -> requires date evaluation
        rule_id = amend_chunk.transitional_rule_id or "§5.1"
        rule_def = self.rules.get(rule_id, self.rules["§5.1"])

        # Check for Period-Spanning evaluation (§5.3)
        if start_date and end_date:
            if start_date < rule_def.effective_threshold <= end_date:
                spanning_rule = self.rules["§5.3"]
                explanation = (
                    f"Clause {cid} spans across the cutover date {rule_def.effective_threshold}. "
                    f"Under transitional rule {spanning_rule.name}, entitlement is apportioned: "
                    f"Base rate applies for period {start_date} to 2026-02-28; "
                    f"Amended rate applies for period 2026-03-01 to {end_date}."
                )
                effective_text = (
                    f"[Apportioned Claim]: "
                    f"Period {start_date} to 2026-02-28: {base_chunk.content} | "
                    f"Period 2026-03-01 to {end_date}: {amend_chunk.content}"
                )
                return ResolvedClause(
                    clause_id=cid,
                    effective_text=effective_text,
                    applied_source="amendment-2026-01",
                    transitional_rule_applied=spanning_rule.name,
                    is_amended=True,
                    explanation=explanation,
                    claim_date=claim_date or start_date,
                    event_date=event_date,
                )

        # Determine effective evaluation date
        eval_date = None
        if rule_id == "§5.2":
            # For §5.2, event_date / assessment_period takes priority, fallback to claim_date
            eval_date = event_date if event_date is not None else claim_date
            if eval_date is None:
                raise DateRequiredError(
                    clause_id=cid,
                    rule_id=rule_id,
                    message=f"Claim date or assessment event date is required to resolve amended clause {cid} under transitional rule {rule_id}."
                )
        else:
            eval_date = claim_date
            if eval_date is None:
                raise DateRequiredError(
                    clause_id=cid,
                    rule_id=rule_id,
                    message=f"Claim date is required to resolve amended clause {cid} under transitional rule {rule_id}."
                )

        # Evaluate against threshold
        if eval_date >= rule_def.effective_threshold:
            # Amended version applies
            if rule_id == "§5.2":
                explanation = (
                    f"Event/claim date ({eval_date.isoformat()}) is on or after 1 March 2026. "
                    f"Transitional rule {rule_def.name} applies the amended rate from {amend_chunk.source_file}."
                )
            else:
                explanation = (
                    f"Claim date ({eval_date.isoformat()}) is on or after {rule_def.effective_threshold.isoformat()}. "
                    f"Transitional rule {rule_def.name} activates the amended provision ({amend_chunk.clause_id})."
                )

            return ResolvedClause(
                clause_id=cid,
                effective_text=amend_chunk.content,
                applied_source="amendment-2026-01",
                transitional_rule_applied=rule_def.name,
                is_amended=True,
                explanation=explanation,
                claim_date=claim_date,
                event_date=event_date,
            )
        else:
            # Base version applies
            if rule_id == "§5.2":
                explanation = (
                    f"Event/claim date ({eval_date.isoformat()}) is prior to 1 March 2026. "
                    f"Under transitional rule {rule_def.name}, the prior base policy rate applies until conclusion of that assessment period."
                )
            else:
                explanation = (
                    f"Claim date ({eval_date.isoformat()}) is prior to {rule_def.effective_threshold.isoformat()}. "
                    f"Under transitional rule {rule_def.name}, the base policy manual text governs."
                )

            return ResolvedClause(
                clause_id=cid,
                effective_text=base_chunk.content,
                applied_source="base",
                transitional_rule_applied=rule_def.name,
                is_amended=False,
                explanation=explanation,
                claim_date=claim_date,
                event_date=event_date,
            )

    def resolve(
        self,
        clause_id: str,
        claim_date: Optional[date] = None,
        event_date: Optional[date] = None
    ) -> ResolvedClause:
        """Alias for resolve_clause."""
        return self.resolve_clause(clause_id=clause_id, claim_date=claim_date, event_date=event_date)

    def resolve_batch(
        self,
        clause_ids: List[str],
        claim_date: Optional[date] = None,
        event_date: Optional[date] = None
    ) -> List[ResolvedClause]:
        """Resolve multiple clauses for a given date context."""
        return [self.resolve_clause(cid, claim_date=claim_date, event_date=event_date) for cid in clause_ids]
