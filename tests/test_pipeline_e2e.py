"""tests/test_pipeline_e2e.py

End-to-End integration tests for 'The Grounded Answer' pipeline.
"""

from datetime import date
import pytest
from main import GroundedAnswerPipeline
from answer_builder import GroundedAnswer
from refusal_gate import RefusalEvaluation, RefusalReason


@pytest.fixture(scope="module")
def pipeline():
    return GroundedAnswerPipeline()


def test_e2e_pre_march_earnings_disregard(pipeline):
    """End-to-end test: Pre-March 1, 2026 inquiry returns base manual rate ($200.00)."""
    result = pipeline.run_query(
        query="What is the weekly earnings disregard allowance?",
        claim_date=date(2026, 2, 15)
    )

    assert isinstance(result, GroundedAnswer)
    assert result.is_refusal is False
    assert "§6.4.1(a)" in result.cited_clauses
    assert "$200.00" in result.answer_text
    assert "2026-02-15" in (result.applied_date_context or "")


def test_e2e_post_march_earnings_disregard(pipeline):
    """End-to-end test: Post-March 1, 2026 inquiry returns amended rate ($260.00) and cites §5.2."""
    result = pipeline.run_query(
        query="What is the weekly earnings disregard allowance?",
        claim_date=date(2026, 3, 15)
    )

    assert isinstance(result, GroundedAnswer)
    assert result.is_refusal is False
    assert "§6.4.1(a)" in result.cited_clauses
    assert "$260.00" in result.answer_text
    assert "§5.2" in (result.transitional_summary or "") or "§5.2" in result.cited_clauses


def test_e2e_travel_lodging_amendment(pipeline):
    """End-to-end test: Lodging inquiry evaluated across cutover date."""
    # Pre-cutover
    pre_res = pipeline.run_query(
        query="What is the daily lodging reimbursement limit?",
        claim_date=date(2026, 1, 20)
    )
    assert isinstance(pre_res, GroundedAnswer)
    assert "$150.00" in pre_res.answer_text
    assert "§1.1.1" in pre_res.cited_clauses

    # Post-cutover
    post_res = pipeline.run_query(
        query="What is the daily lodging reimbursement limit?",
        claim_date=date(2026, 4, 10)
    )
    assert isinstance(post_res, GroundedAnswer)
    assert "$185.00" in post_res.answer_text
    assert "§1.1.1" in post_res.cited_clauses


def test_e2e_out_of_scope_refusal(pipeline):
    """End-to-end test: Out of scope inquiry triggers RefusalEvaluation with contact guidance."""
    result = pipeline.run_query(
        query="What is the company policy regarding bringing dogs and pets into the office?",
        claim_date=date(2026, 3, 1)
    )

    assert isinstance(result, RefusalEvaluation)
    assert result.should_refuse is True
    assert result.reason == RefusalReason.OUT_OF_SCOPE.value
    assert "HR Policy Desk" in result.suggested_contact or "Operations" in result.suggested_contact


def test_e2e_missing_date_on_amended_clause_refusal(pipeline):
    """End-to-end test: Query on amended clause without date returns date requirement refusal."""
    result = pipeline.run_query(
        query="What is the weekly earnings disregard allowance?",
        claim_date=None
    )

    assert isinstance(result, RefusalEvaluation)
    assert result.should_refuse is True
    assert result.reason == RefusalReason.MISSING_DATE_CONTEXT.value
