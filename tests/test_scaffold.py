"""test_scaffold.py

Verification tests for module importability and scaffold interfaces.
"""

from datetime import date
import pytest
from clause_resolver import ClauseResolver, ClauseVersion, ResolvedClause, TransitionalRule
from refusal_gate import RefusalGate, RefusalReason
from answer_builder import AnswerBuilder, Citation, GroundedAnswer
from retriever import RetrievedClause
from ingest import ClauseChunk


def test_module_interfaces_and_types():
    """Verify that all core dataclasses and types instantiate correctly."""
    # Test ClauseChunk
    chunk = ClauseChunk(
        clause_id="§1.1.1",
        title="Domestic Travel Lodging",
        content="Daily lodging capped at $150.00",
        source_file="policy-manual.md"
    )
    assert chunk.clause_id == "§1.1.1"

    # Test TransitionalRule
    rule = TransitionalRule(
        rule_id="TR-1.1.1",
        description="Effective March 1, 2026",
        effective_threshold=date(2026, 3, 1),
        applicability_condition=lambda d: d >= date(2026, 3, 1)
    )
    assert rule.applicability_condition(date(2026, 3, 1)) is True
    assert rule.applicability_condition(date(2026, 2, 28)) is False

    # Test ResolvedClause
    resolved = ResolvedClause(
        clause_id="§1.1.1",
        version_applied=ClauseVersion.AMENDED,
        resolved_text="Daily lodging capped at $185.00",
        transitional_rule_applied="TR-1.1.1",
        claim_date=date(2026, 3, 15),
        source_reference="amendment-2026-01.md"
    )
    assert resolved.version_applied == ClauseVersion.AMENDED

    # Test GroundedAnswer
    citation = Citation(
        clause_id="§1.1.1",
        version="amended",
        transitional_rule="TR-1.1.1",
        source_quote="$185.00 per night"
    )
    answer = GroundedAnswer(
        question="What is lodging limit on March 15, 2026?",
        claim_date=date(2026, 3, 15),
        answer_text="The lodging limit is $185.00 per night.",
        citations=[citation]
    )
    assert len(answer.citations) == 1
    assert answer.citations[0].clause_id == "§1.1.1"
