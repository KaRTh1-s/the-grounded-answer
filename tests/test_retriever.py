"""tests/test_retriever.py

Unit tests for semantic retrieval, exact clause matching, and score threshold filtering.
"""

import pytest
from qdrant_client import QdrantClient
from ingest import ingest_to_qdrant, load_and_chunk_corpus
from retriever import PolicyRetriever, RetrievedClause


@pytest.fixture(scope="module")
def retriever():
    """Create in-memory Qdrant client and populate with corpus for retrieval tests."""
    client = QdrantClient(location=":memory:")
    chunks = load_and_chunk_corpus("corpus")
    ingest_to_qdrant(chunks, client=client, collection_name="policy_corpus")
    return PolicyRetriever(client=client, collection_name="policy_corpus")


def test_retrieve_earnings_disregard(retriever):
    """Query about earnings disregard must retrieve §6.4.1(a)."""
    results = retriever.retrieve("What is the earnings disregard allowance?", top_k=4)
    assert len(results) > 0
    clause_ids = [r.clause_id for r in results]
    assert "§6.4.1(a)" in clause_ids

    # Top result should be related to earnings disregard
    top = results[0]
    assert "6.4.1" in top.clause_id
    assert top.similarity_score >= 0.35


def test_retrieve_outpatient_medical_claims(retriever):
    """Query about outpatient medical claims must retrieve §2.1.1."""
    results = retriever.retrieve("What are the outpatient medical benefits for preventative dental?", top_k=4)
    assert len(results) > 0
    clause_ids = [r.clause_id for r in results]
    assert "§2.1.1" in clause_ids


def test_exact_clause_id_query_prioritization(retriever):
    """Query explicitly mentioning '§4.3.1' must boost that clause to top rank with max score."""
    results = retriever.retrieve("What does clause §4.3.1 say about deadlines?", top_k=3)
    assert len(results) > 0
    top = results[0]
    assert top.clause_id == "§4.3.1"
    assert top.similarity_score == 1.0


def test_low_relevance_query_filtering(retriever):
    """Unrelated query must return empty list or scores below strict threshold."""
    results = retriever.retrieve("How do I bake a chocolate cake with frosting?", score_threshold=0.6)
    assert len(results) == 0


def test_retrieved_clause_structure(retriever):
    """Verify all fields and properties on RetrievedClause instances."""
    results = retriever.retrieve("lodging reimbursement", top_k=1)
    assert len(results) == 1
    rc = results[0]
    assert isinstance(rc.clause_id, str)
    assert isinstance(rc.text, str)
    assert isinstance(rc.similarity_score, float)
    assert rc.similarity_score > 0
    assert hasattr(rc, "part_title")
    assert hasattr(rc, "section_title")
