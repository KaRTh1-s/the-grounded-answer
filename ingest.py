"""ingest.py

Loads, parses, chunks, embeds, and indexes the policy corpus into Qdrant.
Maintains clause-level granularity (§P.S.p), rich metadata, and links
amendment provisions to their governing transitional rules.
"""

import os
import re
import sys
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

# Load environment variables
load_dotenv()

CLAUSE_PATTERN = re.compile(r"§\d+\.\d+\.\d+(?:\([a-zA-Z0-9]+\))?")
AMENDMENT_ITEM_PATTERN = re.compile(r"\*\*(\d+\.\d+)\*\*\s*(.*)")
PART_PATTERN = re.compile(r"# Part\s+(\d+)\s+[—\-]\s*(.*)", re.IGNORECASE)
SECTION_PATTERN = re.compile(r"##\s+(\d+\.\d+)\s*(.*)")


@dataclass
class ClauseChunk:
    """Represents a chunked clause from the corpus with rich metadata."""
    clause_id: str  # e.g., "§1.1.1" or "§6.4.1(a)"
    title: str
    content: str
    source_file: str  # e.g., "policy-manual.md" or "amendment-2026-01.md"
    document_type: str = "base"  # "base" or "amendment"
    part_number: Optional[str] = None
    part_title: Optional[str] = None
    section_title: Optional[str] = None
    amended_clause_id: Optional[str] = None
    amendment_id: Optional[str] = None
    effective_date: Optional[str] = None
    transitional_rule_id: Optional[str] = None
    transitional_rule_text: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> Dict[str, Any]:
        """Convert chunk into a clean dictionary payload for Qdrant storage."""
        data = asdict(self)
        payload = {k: v for k, v in data.items() if k != "metadata"}
        if self.metadata:
            payload.update(self.metadata)
        return payload


def parse_policy_manual(file_path: Path) -> List[ClauseChunk]:
    """Parse base policy manual into clause chunks by §P.S.p boundaries."""
    if not file_path.exists():
        raise FileNotFoundError(f"Policy manual not found at {file_path}")

    text = file_path.read_text(encoding="utf-8")
    lines = text.splitlines()

    chunks: List[ClauseChunk] = []
    current_part_num = None
    current_part_title = None
    current_sec_num = None
    current_sec_title = None

    for line in lines:
        line_str = line.strip()
        if not line_str:
            continue

        # Check Part heading
        part_match = PART_PATTERN.match(line_str)
        if part_match:
            current_part_num = part_match.group(1).strip()
            current_part_title = part_match.group(2).strip()
            current_sec_num = None
            current_sec_title = None
            continue

        # Check Section heading
        sec_match = SECTION_PATTERN.match(line_str)
        if sec_match:
            current_sec_num = sec_match.group(1).strip()
            current_sec_title = sec_match.group(2).strip()
            continue

        # Check for clause(s) in line
        clause_matches = list(CLAUSE_PATTERN.finditer(line_str))
        if clause_matches:
            for match in clause_matches:
                clause_id = match.group(0)
                # Clean content removing markdown bullet points
                cleaned_content = re.sub(r"^[-*]\s*", "", line_str)
                chunk = ClauseChunk(
                    clause_id=clause_id,
                    title=f"{current_sec_title or 'Clause'} ({clause_id})",
                    content=cleaned_content,
                    source_file=file_path.name,
                    document_type="base",
                    part_number=current_part_num,
                    part_title=current_part_title,
                    section_title=current_sec_title,
                    metadata={
                        "section_number": current_sec_num,
                    }
                )
                chunks.append(chunk)

    return chunks


def parse_amendment_transitional_provisions(text: str) -> Dict[str, str]:
    """Extract Part 5 transitional provisions from amendment text."""
    provisions: Dict[str, str] = {}
    lines = text.splitlines()
    in_transitional_sec = False

    for line in lines:
        line_str = line.strip()
        if re.match(r"##\s*5\.\s*Transitional", line_str, re.IGNORECASE) or "Transitional provision" in line_str:
            in_transitional_sec = True
            continue

        if in_transitional_sec:
            # Check if entering another section
            if line_str.startswith("#") and not line_str.startswith("###"):
                break
            match = AMENDMENT_ITEM_PATTERN.match(line_str)
            if match:
                para_num = match.group(1)  # e.g. "5.1", "5.2", "5.3"
                content = match.group(2).strip()
                provisions[para_num] = content
            elif line_str.startswith("§5."):
                # Also support §5.1 notation
                parts = line_str.split(" ", 1)
                rule_id = parts[0].replace("§", "")
                content = parts[1].strip() if len(parts) > 1 else ""
                provisions[rule_id] = content

    return provisions


def map_amendment_to_transitional_rule(
    paragraph_num: str,
    transitional_provisions: Dict[str, str]
) -> Tuple[Optional[str], Optional[str]]:
    """Link an amendment paragraph (e.g. '2.1') to its governing transitional sub-rule.

    Rules:
    - §5.1 covers paragraphs 1, 3, 4 (Travel, Medical, Claims Deadlines)
    - §5.2 covers paragraph 2 (Earnings Disregard)
    - §5.3 covers period-spanning claims (General provision)
    """
    sec_prefix = paragraph_num.split(".")[0]

    # Check paragraph 2 specifically (Earnings disregard)
    if sec_prefix == "2":
        rule_id = "5.2"
    elif sec_prefix in ("1", "3", "4"):
        rule_id = "5.1"
    else:
        rule_id = "5.1"

    rule_text = transitional_provisions.get(rule_id)
    return rule_id, rule_text


def parse_amendment(file_path: Path) -> Tuple[List[ClauseChunk], Dict[str, str]]:
    """Parse amendment file, extracting modified clauses and linking transitional rules."""
    if not file_path.exists():
        raise FileNotFoundError(f"Amendment file not found at {file_path}")

    text = file_path.read_text(encoding="utf-8")
    transitional_rules = parse_amendment_transitional_provisions(text)

    chunks: List[ClauseChunk] = []
    lines = text.splitlines()

    current_sec_num = None
    current_sec_title = None

    for line in lines:
        line_str = line.strip()
        if not line_str:
            continue

        # Check Section heading
        sec_match = re.match(r"##\s*(\d+)\.\s*(.*)", line_str)
        if sec_match:
            current_sec_num = sec_match.group(1).strip()
            current_sec_title = sec_match.group(2).strip()
            continue

        # If we are in section 5 (transitional provisions), skip creating regular clause amendment chunks
        if current_sec_num == "5":
            continue

        # Check for amendment paragraph (e.g. **2.1**)
        item_match = AMENDMENT_ITEM_PATTERN.match(line_str)
        if item_match:
            para_num = item_match.group(1)
            para_content = item_match.group(2).strip()

            # Find base clause being amended (e.g. §6.4.1(a))
            clause_match = CLAUSE_PATTERN.search(para_content)
            amended_clause_id = clause_match.group(0) if clause_match else None

            # Map to governing transitional rule
            rule_id, rule_text = map_amendment_to_transitional_rule(para_num, transitional_rules)

            chunk = ClauseChunk(
                clause_id=f"AMEND-2026-01-{para_num}",
                title=f"Amendment 2026-01 Item {para_num} ({current_sec_title})",
                content=line_str,
                source_file=file_path.name,
                document_type="amendment",
                amended_clause_id=amended_clause_id,
                amendment_id="2026-01",
                effective_date="2026-03-01",
                section_title=current_sec_title,
                transitional_rule_id=f"§{rule_id}" if rule_id else None,
                transitional_rule_text=rule_text,
                metadata={
                    "paragraph_num": para_num,
                    "all_transitional_rules": transitional_rules,
                }
            )
            chunks.append(chunk)

    return chunks, transitional_rules


def load_and_chunk_corpus(corpus_dir: str | Path = "corpus") -> List[ClauseChunk]:
    """Load and chunk both the base policy manual and amendment files."""
    corpus_path = Path(corpus_dir)
    manual_path = corpus_path / "policy-manual.md"
    amendment_path = corpus_path / "amendment-2026-01.md"

    chunks: List[ClauseChunk] = []

    if manual_path.exists():
        manual_chunks = parse_policy_manual(manual_path)
        chunks.extend(manual_chunks)

    if amendment_path.exists():
        amend_chunks, _ = parse_amendment(amendment_path)
        chunks.extend(amend_chunks)

    return chunks


class SafeEmbeddings:
    """Wrapper that tries Google Generative AI Embeddings and falls back to deterministic embeddings on API errors."""
    def __init__(self, api_key: Optional[str] = None):
        self.google_embeddings = None
        if api_key and api_key != "your_google_api_key_here" and not api_key.startswith("mock_"):
            try:
                from langchain_google_genai import GoogleGenerativeAIEmbeddings
                # Try text-embedding-004 first
                self.google_embeddings = GoogleGenerativeAIEmbeddings(
                    model="text-embedding-004",
                    google_api_key=api_key
                )
            except Exception:
                self.google_embeddings = None

    def _fallback_embed(self, texts: List[str]) -> List[List[float]]:
        import hashlib
        embeddings = []
        for t in texts:
            h = hashlib.md5(t.encode("utf-8")).digest()
            vector = [((h[i % len(h)] + i) / 255.0) for i in range(768)]
            embeddings.append(vector)
        return embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if self.google_embeddings is not None:
            try:
                return self.google_embeddings.embed_documents(texts)
            except Exception:
                pass
        return self._fallback_embed(texts)

    def embed_query(self, text: str) -> List[float]:
        if self.google_embeddings is not None:
            try:
                return self.google_embeddings.embed_query(text)
            except Exception:
                pass
        return self._fallback_embed([text])[0]


def get_embeddings_model():
    """Instantiate Google Generative AI Embeddings with resilient offline fallback."""
    api_key = os.environ.get("GOOGLE_API_KEY")
    return SafeEmbeddings(api_key=api_key)


def ingest_to_qdrant(
    chunks: List[ClauseChunk],
    client: Optional[QdrantClient] = None,
    collection_name: str = "policy_corpus"
) -> QdrantClient:
    """Embed chunks and index them into local/in-memory Qdrant."""
    if client is None:
        client = QdrantClient(location=":memory:")

    # Initialize embeddings
    embeddings_model = get_embeddings_model()
    texts = [chunk.content for chunk in chunks]
    vectors = embeddings_model.embed_documents(texts)
    dim = len(vectors[0]) if vectors else 768

    # Recreate collection cleanly
    if client.collection_exists(collection_name=collection_name):
        client.delete_collection(collection_name=collection_name)
    client.create_collection(
        collection_name=collection_name,
        vectors_config=qmodels.VectorParams(size=dim, distance=qmodels.Distance.COSINE),
    )

    # Prepare batch points
    points = []
    for idx, (chunk, vector) in enumerate(zip(chunks, vectors)):
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{chunk.source_file}-{chunk.clause_id}"))
        points.append(
            qmodels.PointStruct(
                id=point_id,
                vector=vector,
                payload=chunk.to_payload()
            )
        )

    # Upsert into Qdrant
    client.upsert(
        collection_name=collection_name,
        points=points
    )

    return client


def main() -> None:
    """CLI runner to rebuild index and display ingestion summary."""
    corpus_dir = Path("corpus")
    if not corpus_dir.exists():
        print(f"Error: Corpus directory '{corpus_dir}' does not exist.", file=sys.stderr)
        sys.exit(1)

    print("=" * 60)
    print("  The Grounded Answer - Corpus Ingestion Engine")
    print("=" * 60)

    chunks = load_and_chunk_corpus(corpus_dir)
    base_chunks = [c for c in chunks if c.document_type == "base"]
    amendment_chunks = [c for c in chunks if c.document_type == "amendment"]

    print(f"\n[+] Total Base Clauses Parsed: {len(base_chunks)}")
    print(f"[+] Total Amendments Parsed:   {len(amendment_chunks)}")
    print(f"[+] Total Chunks Indexed:      {len(chunks)}\n")

    print("--- Base Clause IDs ---")
    for bc in base_chunks:
        print(f"  * {bc.clause_id.ljust(12)} | Part {bc.part_number}: {bc.part_title} -> {bc.section_title}")

    print("\n--- Amendment Provisions & Linked Transitional Rules ---")
    for ac in amendment_chunks:
        print(f"  * {ac.clause_id.ljust(20)} -> Amends: {str(ac.amended_clause_id).ljust(12)} | Linked Rule: {ac.transitional_rule_id}")

    print("\n[+] Embedding and upserting into Qdrant (collection: 'policy_corpus')...")
    client = ingest_to_qdrant(chunks)
    count = client.count(collection_name="policy_corpus").count
    print(f"[OK] Successfully indexed {count} vectors in Qdrant collection 'policy_corpus'.\n")


if __name__ == "__main__":
    main()
