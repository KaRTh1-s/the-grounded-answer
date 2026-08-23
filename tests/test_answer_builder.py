"""tests/test_answer_builder.py

Unit tests for grounded answer building, citation tracking, and transitional explanations.
"""

from datetime import date
import pytest
from answer_builder import AnswerBuilder, GroundedAnswer
from clause_resolver import ResolvedClause


@pytest.fixture
def builder():
    return AnswerBuilder()


def test_build_answer_with_clause_citation(builder):
    """Answer builder must produce an answer that cites the source clause ID."""
    resolved = [
        ResolvedClause(
            clause_id="§6.4.1(a)",
            effective_text="The standard weekly earnings disregard is $260.00 per week.",
            applied_source="amendment-2026-01",
            transitional_rule_applied="§5.2 Event-based Effective Date",
            is_amended=True,
            explanation="Claim date after 1 March 2026 applies amended rate.",
            claim_date=date(2026, 3, 15)
        )
    ]

    answer = builder.build_answer(
        query="What is the weekly earnings disregard?",
        resolved_clauses=resolved,
        claim_date=date(2026, 3, 15)
    )

    assert isinstance(answer, GroundedAnswer)
    assert answer.is_refusal is False
    assert "§6.4.1(a)" in answer.cited_clauses
    assert "§6.4.1(a)" in answer.answer_text
    assert answer.applied_date_context == "2026-03-15"
    assert answer.transitional_summary is not None
    assert "§5.2" in answer.transitional_summary


def test_build_answer_base_unamended_clause(builder):
    """Answer builder handles unamended clauses with clean citation to base manual."""
    resolved = [
        ResolvedClause(
            clause_id="§1.1.2",
            effective_text="Daily meal reimbursement rate is capped at $45.00 per day.",
            applied_source="base",
            transitional_rule_applied="None (Unamended)",
            is_amended=False,
            explanation="Base manual text applies.",
            claim_date=date(2026, 1, 10)
        )
    ]

    answer = builder.build_answer(
        query="What is the daily meal reimbursement limit?",
        resolved_clauses=resolved,
        claim_date=date(2026, 1, 10)
    )

    assert "§1.1.2" in answer.cited_clauses
    assert "$45.00" in answer.answer_text
    assert len(answer.citations) == 1
    assert answer.citations[0].clause_id == "§1.1.2"


def test_build_answer_empty_resolved_clauses(builder):
    """Empty resolved clauses returns refusal flag."""
    answer = builder.build_answer(
        query="What is the travel policy?",
        resolved_clauses=[],
        claim_date=date(2026, 3, 1)
    )
    assert answer.is_refusal is True
    assert len(answer.cited_clauses) == 0
