"""tests/test_ingest.py

Unit tests for corpus parsing, chunking, amendment linking, and Qdrant ingestion.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
from ingest import (
    ClauseChunk,
    ingest_to_qdrant,
    load_and_chunk_corpus,
    map_amendment_to_transitional_rule,
    parse_amendment,
    parse_amendment_transitional_provisions,
    parse_policy_manual,
)


@pytest.fixture
def corpus_dir():
    return Path("corpus")


def test_base_manual_parses_clause_6_4_1_a(corpus_dir):
    """Verify that clause §6.4.1(a) is correctly extracted from the base manual."""
    manual_path = corpus_dir / "policy-manual.md"
    chunks = parse_policy_manual(manual_path)

    clause_ids = [c.clause_id for c in chunks]
    assert "§6.4.1(a)" in clause_ids

    chunk_641a = next(c for c in chunks if c.clause_id == "§6.4.1(a)")
    assert chunk_641a.document_type == "base"
    assert chunk_641a.part_number == "6"
    assert "Income & Earnings Disregards" in (chunk_641a.part_title or "")
    assert "$200.00 per week" in chunk_641a.content


def test_amendment_earnings_disregard_links_to_transitional_rule(corpus_dir):
    """Verify that amendment paragraph 2.1 (earnings disregard) links to §5.2 transitional rule."""
    amend_path = corpus_dir / "amendment-2026-01.md"
    chunks, transitional_rules = parse_amendment(amend_path)

    assert "5.2" in transitional_rules
    assert "assessment periods" in transitional_rules["5.2"]

    # Find amendment chunk for item 2.1
    chunk_21 = next(c for c in chunks if c.clause_id == "AMEND-2026-01-2.1")
    assert chunk_21.amended_clause_id == "§6.4.1(a)"
    assert chunk_21.transitional_rule_id == "§5.2"
    assert chunk_21.transitional_rule_text is not None
    assert "Paragraph 2 (earnings disregard)" in chunk_21.transitional_rule_text


def test_total_chunk_count_and_accuracy(corpus_dir):
    """Verify total chunk count is non-zero and accurately combines base and amendment chunks."""
    chunks = load_and_chunk_corpus(corpus_dir)
    assert len(chunks) > 0

    base_chunks = [c for c in chunks if c.document_type == "base"]
    amend_chunks = [c for c in chunks if c.document_type == "amendment"]

    assert len(base_chunks) >= 10
    assert len(amend_chunks) >= 5

    # Check that each chunk has a valid clause_id and content
    for chunk in chunks:
        assert chunk.clause_id.startswith("§") or chunk.clause_id.startswith("AMEND-")
        assert len(chunk.content.strip()) > 0
        assert chunk.source_file in ["policy-manual.md", "amendment-2026-01.md"]


def test_ingest_to_qdrant_runs_cleanly(corpus_dir):
    """Verify Qdrant ingestion creates collection and indexes vectors even without API key."""
    chunks = load_and_chunk_corpus(corpus_dir)
    client = ingest_to_qdrant(chunks)

    count_result = client.count(collection_name="policy_corpus")
    assert count_result.count == len(chunks)

    # Verify payload integrity for a retrieved point
    points, _ = client.scroll(
        collection_name="policy_corpus",
        limit=1,
        with_payload=True
    )
    assert len(points) == 1
    payload = points[0].payload
    assert "clause_id" in payload
    assert "content" in payload
    assert "document_type" in payload
