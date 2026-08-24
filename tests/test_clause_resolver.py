"""tests/test_clause_resolver.py

Unit tests for date-aware clause resolution and transitional rule evaluation.
"""

from datetime import date
import pytest
from clause_resolver import (
    ClauseResolver,
    DateRequiredError,
    ResolvedClause,
    TRANSITIONAL_RULES,
)


@pytest.fixture
def resolver():
    return ClauseResolver()


def test_unamended_clause_always_returns_base_text(resolver):
    """Unamended clauses like §1.1.2 must always return base manual text regardless of date."""
    # Even without a claim date
    res_no_date = resolver.resolve_clause("§1.1.2")
    assert res_no_date.applied_source == "base"
    assert res_no_date.is_amended is False
    assert "$45.00 per day" in res_no_date.effective_text

    # With pre-amendment date
    res_pre = resolver.resolve_clause("§1.1.2", claim_date=date(2026, 1, 15))
    assert res_pre.applied_source == "base"
    assert "$45.00 per day" in res_pre.effective_text

    # With post-amendment date
    res_post = resolver.resolve_clause("§1.1.2", claim_date=date(2026, 4, 1))
    assert res_post.applied_source == "base"
    assert "$45.00 per day" in res_post.effective_text


def test_earnings_disregard_pre_march_returns_base(resolver):
    """§6.4.1(a) with claim date before March 1, 2026 returns base manual text ($200.00)."""
    pre_date = date(2026, 2, 28)
    res = resolver.resolve_clause("§6.4.1(a)", claim_date=pre_date)

    assert res.applied_source == "base"
    assert res.is_amended is False
    assert "$200.00 per week" in res.effective_text
    assert "§5.2" in res.transitional_rule_applied
    assert "prior base policy rate applies" in res.explanation


def test_earnings_disregard_post_march_returns_amended(resolver):
    """§6.4.1(a) with claim date on/after March 1, 2026 returns amended text citing rule §5.2."""
    post_date = date(2026, 3, 1)
    res = resolver.resolve_clause("§6.4.1(a)", claim_date=post_date)

    assert res.applied_source == "amendment-2026-01"
    assert res.is_amended is True
    assert "$260.00 per week" in res.effective_text
    assert "§5.2" in res.transitional_rule_applied
    assert "Transitional rule" in res.explanation


def test_rule_5_1_standard_travel_allowance(resolver):
    """§1.1.1 (lodging allowance) under rule §5.1 cutover on March 1, 2026."""
    # Pre-cutover
    res_pre = resolver.resolve_clause("§1.1.1", claim_date=date(2026, 2, 20))
    assert res_pre.applied_source == "base"
    assert "$150.00 per night" in res_pre.effective_text
    assert "§5.1" in res_pre.transitional_rule_applied

    # Post-cutover
    res_post = resolver.resolve_clause("§1.1.1", claim_date=date(2026, 3, 10))
    assert res_post.applied_source == "amendment-2026-01"
    assert "$185.00 per night" in res_post.effective_text
    assert "§5.1" in res_post.transitional_rule_applied


def test_period_spanning_claims_rule_5_3(resolver):
    """Period-spanning claim evaluates under rule §5.3 apportionment logic."""
    start = date(2026, 2, 15)
    end = date(2026, 3, 15)
    res = resolver.resolve_clause("§1.1.1", start_date=start, end_date=end)

    assert res.applied_source == "amendment-2026-01"
    assert res.is_amended is True
    assert "§5.3" in res.transitional_rule_applied
    assert "apportioned" in res.explanation.lower()
    assert "$150.00" in res.effective_text
    assert "$185.00" in res.effective_text


def test_period_spanning_mileage_allowance_rule_5_3(resolver):
    """Period-spanning claim for mileage allowance §1.1.3 calculates pro-rata rates across cutover."""
    start = date(2026, 2, 20)
    end = date(2026, 3, 10)
    res = resolver.resolve_clause("§1.1.3", start_date=start, end_date=end)

    assert res.applied_source == "amendment-2026-01"
    assert res.is_amended is True
    assert "§5.3" in res.transitional_rule_applied
    assert "apportioned" in res.explanation.lower()
    assert "$0.58" in res.effective_text
    assert "$0.67" in res.effective_text


def test_period_spanning_entirely_pre_cutover(resolver):
    """Date span entirely before 1 March 2026 evaluates to pure base rate without apportionment."""
    start = date(2026, 2, 1)
    end = date(2026, 2, 25)
    res = resolver.resolve_clause("§1.1.1", claim_date=start, start_date=start, end_date=end)

    assert res.applied_source == "base"
    assert res.is_amended is False
    assert "$150.00 per night" in res.effective_text
    assert "§5.3" not in res.transitional_rule_applied


def test_period_spanning_entirely_post_cutover(resolver):
    """Date span entirely on/after 1 March 2026 evaluates to pure amended rate without apportionment."""
    start = date(2026, 3, 5)
    end = date(2026, 3, 25)
    res = resolver.resolve_clause("§1.1.1", claim_date=start, start_date=start, end_date=end)

    assert res.applied_source == "amendment-2026-01"
    assert res.is_amended is True
    assert "$185.00 per night" in res.effective_text
    assert "§5.1" in res.transitional_rule_applied


def test_missing_date_for_amended_clause_raises_error(resolver):
    """Omission of claim_date for an amended clause raises DateRequiredError."""
    with pytest.raises(DateRequiredError) as exc_info:
        resolver.resolve_clause("§6.4.1(a)")

    assert exc_info.value.clause_id == "§6.4.1(a)"
    assert exc_info.value.rule_id == "§5.2"


def test_invalid_clause_id_raises_keyerror(resolver):
    """Non-existent clause ID raises KeyError."""
    with pytest.raises(KeyError):
        resolver.resolve_clause("§9.9.9", claim_date=date(2026, 3, 1))
