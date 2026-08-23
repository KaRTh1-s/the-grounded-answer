"""retriever.py

Semantic retriever and vector search interface with local Qdrant.
Performs similarity search, exact clause pattern matching, deduplication,
and score filtering.
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from ingest import get_embeddings_model, ingest_to_qdrant, load_and_chunk_corpus

CLAUSE_QUERY_PATTERN = re.compile(
    r"(?:§|section|clause|item)?\s*(\d+\.\d+\.\d+(?:\([a-zA-Z0-9]+\))?)",
    re.IGNORECASE
)


@dataclass
class RetrievedClause:
    """A clause candidate returned from vector similarity and keyword search."""
    clause_id: str
    text: str
    similarity_score: float
    part_title: str = ""
    section_title: str = ""
    document_type: str = "base"  # "base" | "amendment"
    transitional_rule_id: Optional[str] = None
    source_file: str = "policy-manual.md"
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Backward compatibility properties
    @property
    def content(self) -> str:
        return self.text

    @property
    def score(self) -> float:
        return self.similarity_score


class PolicyRetriever:
    """Interface to query local Qdrant vector collection for relevant policy clauses."""

    def __init__(
        self,
        client: Optional[QdrantClient] = None,
        collection_name: str = "policy_corpus",
        embeddings_model: Any = None
    ):
        self.collection_name = collection_name
        self.embeddings_model = embeddings_model or get_embeddings_model()

        if client is None:
            # Auto-initialize in-memory store if not provided
            self.client = QdrantClient(location=":memory:")
            chunks = load_and_chunk_corpus()
            ingest_to_qdrant(chunks, client=self.client, collection_name=self.collection_name)
        else:
            self.client = client

    def _extract_exact_clause_mention(self, query: str) -> Optional[str]:
        """Extract explicit clause ID from query text if present (e.g. '§4.3.1', '6.4.1(a)')."""
        match = CLAUSE_QUERY_PATTERN.search(query)
        if match:
            extracted = match.group(1).strip()
            return f"§{extracted}" if not extracted.startswith("§") else extracted
        return None

    def _compute_keyword_boost(self, query: str, payload: Dict[str, Any]) -> float:
        """Compute keyword overlap boost for hybrid ranking."""
        query_words = set(re.findall(r"\w+", query.lower()))
        # Filter out common stop words
        stop_words = {"what", "is", "the", "for", "a", "an", "in", "of", "to", "on", "and", "does", "say", "about", "how", "much"}
        meaningful_words = query_words - stop_words
        if not meaningful_words:
            return 0.0

        content_text = f"{payload.get('title', '')} {payload.get('part_title', '')} {payload.get('section_title', '')} {payload.get('content', '')}".lower()
        content_words = set(re.findall(r"\w+", content_text))

        matches = meaningful_words.intersection(content_words)
        return len(matches) / len(meaningful_words)

    def retrieve(
        self,
        query: str,
        top_k: int = 4,
        score_threshold: float = 0.35
    ) -> List[RetrievedClause]:
        """Query vector store and return candidate clauses ranked by relevance.

        Args:
            query: User's natural language question.
            top_k: Maximum number of clauses to return.
            score_threshold: Minimum similarity cutoff score.

        Returns:
            Ranked, deduplicated list of RetrievedClause instances.
        """
        normalized_query = query.strip()
        if not normalized_query:
            return []

        exact_clause = self._extract_exact_clause_mention(normalized_query)

        # Check if collection exists
        if not self.client.collection_exists(self.collection_name):
            return []

        # Get all points in collection for hybrid scoring
        all_points, _ = self.client.scroll(
            collection_name=self.collection_name,
            limit=200,
            with_payload=True,
            with_vectors=True
        )

        # Generate query embedding
        query_vector = self.embeddings_model.embed_query(normalized_query)

        # Compute cosine similarity helper
        def cosine_similarity(v1: List[float], v2: List[float]) -> float:
            dot = sum(a * b for a, b in zip(v1, v2))
            norm1 = sum(a * a for a in v1) ** 0.5
            norm2 = sum(b * b for b in v2) ** 0.5
            if norm1 == 0 or norm2 == 0:
                return 0.0
            return dot / (norm1 * norm2)

        candidates: List[RetrievedClause] = []

        for point in all_points:
            payload = point.payload or {}
            clause_id = payload.get("clause_id", "")
            doc_type = payload.get("document_type", "base")
            amended_clause_id = payload.get("amended_clause_id")

            # Determine primary clause key for deduplication
            primary_key = amended_clause_id if doc_type == "amendment" and amended_clause_id else clause_id

            # Cosine similarity
            point_vec = point.vector if isinstance(point.vector, list) else []
            vec_sim = cosine_similarity(query_vector, point_vec) if point_vec else 0.0
            # Normalize cosine similarity to [0, 1] range if needed
            vec_sim = max(0.0, min(1.0, (vec_sim + 1.0) / 2.0 if vec_sim < 0 else vec_sim))

            keyword_score = self._compute_keyword_boost(normalized_query, payload)

            # Combined hybrid score
            hybrid_score = (keyword_score * 0.7) + (vec_sim * 0.3) if keyword_score > 0 else (vec_sim * 0.5)

            # Prioritize exact clause mention
            if exact_clause and (clause_id == exact_clause or amended_clause_id == exact_clause):
                hybrid_score = 1.0

            if hybrid_score < score_threshold and not (exact_clause and (clause_id == exact_clause or amended_clause_id == exact_clause)):
                continue

            retrieved = RetrievedClause(
                clause_id=primary_key or clause_id,
                text=payload.get("content", ""),
                similarity_score=round(hybrid_score, 4),
                part_title=payload.get("part_title") or "",
                section_title=payload.get("section_title") or "",
                document_type=doc_type,
                transitional_rule_id=payload.get("transitional_rule_id"),
                source_file=payload.get("source_file", "policy-manual.md"),
                metadata=payload
            )
            candidates.append(retrieved)

        # Sort by similarity score descending
        candidates.sort(key=lambda x: x.similarity_score, reverse=True)

        # Deduplicate by clause_id preserving highest score
        deduped: List[RetrievedClause] = []
        seen_clauses = set()
        for cand in candidates:
            if cand.clause_id not in seen_clauses:
                seen_clauses.add(cand.clause_id)
                deduped.append(cand)
            if len(deduped) >= top_k:
                break

        return deduped


# Alias for backward compatibility
Retriever = PolicyRetriever
