"""tests/test_refusal_gate.py

Unit tests for refusal gate, out-of-scope detection, low-confidence filtering,
and contact routing.
"""

from datetime import date
import pytest
from clause_resolver import ResolvedClause
from refusal_gate import RefusalEvaluation, RefusalGate, RefusalReason
from retriever import RetrievedClause


@pytest.fixture
def gate():
    return RefusalGate(confidence_threshold=0.40)


def test_out_of_scope_query_refusal(gate):
    """Query about unsupported topics like pets/parking must be refused with OUT_OF_SCOPE."""
    retrieved = []
    resolved = []
    eval_res = gate.evaluate("What is the company pet allowance for office visits?", retrieved, resolved)

    assert eval_res.should_refuse is True
    assert eval_res.reason == RefusalReason.OUT_OF_SCOPE.value
    assert "Policy Manual and Amendments do not cover this topic" in eval_res.message
    assert eval_res.suggested_contact is not None


def test_low_confidence_retrieval_refusal(gate):
    """Query with low similarity scores (< threshold) must be refused with LOW_CONFIDENCE."""
    low_clause = RetrievedClause(
        clause_id="§1.1.1",
        text="Standard daily lodging allowance",
        similarity_score=0.15,
        part_title="Travel",
        section_title="Lodging"
    )
    eval_res = gate.evaluate("What is the company policy on space travel?", [low_clause], [])

    assert eval_res.should_refuse is True
    assert eval_res.reason == RefusalReason.LOW_CONFIDENCE.value
    assert "Retrieval confidence is insufficient" in eval_res.message


def test_valid_policy_query_passes(gate):
    """Query with high similarity and resolved clauses must pass refusal gate (should_refuse=False)."""
    retrieved = [
        RetrievedClause(
            clause_id="§6.4.1(a)",
            text="The standard weekly earnings disregard is $260.00",
            similarity_score=0.85,
            part_title="Income & Earnings Disregards",
            section_title="Standard Allowances & Disregards"
        )
    ]
    resolved = [
        ResolvedClause(
            clause_id="§6.4.1(a)",
            effective_text="The standard weekly earnings disregard is $260.00",
            applied_source="amendment-2026-01",
            transitional_rule_applied="§5.2 Event-based Effective Date",
            is_amended=True,
            explanation="Claim date after March 1, 2026",
            claim_date=date(2026, 3, 15)
        )
    ]

    eval_res = gate.evaluate("What is the earnings disregard after March 1?", retrieved, resolved)

    assert eval_res.should_refuse is False
    assert eval_res.reason is None
    assert "successfully validated" in eval_res.message
    assert "Benefits Compliance Team" in eval_res.suggested_contact


def test_contact_routing_for_travel_domain(gate):
    """Refusal on travel questions routes contact to Travel Operations."""
    eval_res = gate.evaluate("What is the pet policy on business flights?", [], [])
    assert eval_res.should_refuse is True
    assert "Travel Operations" in eval_res.suggested_contact


def test_contradiction_detection(gate):
    """Directly contradictory active clauses trigger CONTRADICTION refusal."""
    c1 = ResolvedClause(
        clause_id="§1.1.1",
        effective_text="Direct flight upgrades are strictly prohibited.",
        applied_source="base",
        transitional_rule_applied="None",
        is_amended=False,
        explanation="Base rule"
    )
    c2 = ResolvedClause(
        clause_id="§1.1.1",
        effective_text="Direct flight upgrades are mandatory for all staff.",
        applied_source="base",
        transitional_rule_applied="None",
        is_amended=False,
        explanation="Conflicting rule"
    )
    retrieved = [
        RetrievedClause(
            clause_id="§1.1.1",
            text="Upgrades",
            similarity_score=0.9
        )
    ]
    eval_res = gate.evaluate("Can I get an upgrade?", retrieved, [c1, c2])
    assert eval_res.should_refuse is True
    assert eval_res.reason == RefusalReason.CONTRADICTION.value
