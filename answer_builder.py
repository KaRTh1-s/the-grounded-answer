"""answer_builder.py

Synthesizes grounded, legally binding policy answers using Google Gemini (gemini-2.5-flash)
and strict clause-level citation traceability (§x.x.x).
"""

import os
import re
import logging
import warnings

# Suppress non-critical Google GenAI / LangChain SDK warnings and logs
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*automatic function calling.*")
logging.getLogger("google").setLevel(logging.ERROR)
logging.getLogger("google_genai").setLevel(logging.ERROR)
logging.getLogger("google_genai.models").setLevel(logging.ERROR)
logging.getLogger("google_genai._api_client").setLevel(logging.ERROR)
logging.getLogger("langchain_google_genai").setLevel(logging.ERROR)
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from clause_resolver import ResolvedClause

load_dotenv()


@dataclass
class Citation:
    """Explicit citation linking an answer claim to a specific clause and rule."""
    clause_id: str
    version: str
    transitional_rule: Optional[str]
    source_quote: str


@dataclass
class GroundedAnswer:
    """Complete grounded answer with citation provenance and date context."""
    answer_text: str
    cited_clauses: List[str] = field(default_factory=list)
    is_refusal: bool = False
    applied_date_context: Optional[str] = None
    transitional_summary: Optional[str] = None
    question: str = ""
    claim_date: Optional[date] = None
    citations: List[Citation] = field(default_factory=list)
    confidence: float = 1.0


class AnswerBuilder:
    """Constructs verifiable, grounded answers from date-resolved clauses using Gemini."""

    def __init__(self, model_name: str = "gemini-2.5-flash"):
        self.model_name = model_name
        self.llm = self._init_llm()

    def _init_llm(self) -> Any:
        """Initialize Google Gemini LLM via LangChain if valid API key is present."""
        api_key = os.environ.get("GOOGLE_API_KEY")
        if api_key and api_key != "your_google_api_key_here" and not api_key.startswith("mock_"):
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
                return ChatGoogleGenerativeAI(
                    model=self.model_name,
                    google_api_key=api_key,
                    temperature=0.0
                )
            except Exception:
                return None
        return None

    def _build_context_block(self, resolved_clauses: List[ResolvedClause]) -> str:
        """Format resolved clauses into structured context for generation."""
        blocks = []
        for rc in resolved_clauses:
            status = f"Amended via {rc.transitional_rule_applied}" if rc.is_amended else "Base Policy"
            block = (
                f"Clause ID: {rc.clause_id}\n"
                f"Status: {status}\n"
                f"Legally Binding Text: {rc.effective_text}\n"
                f"Rationale: {rc.explanation}"
            )
            blocks.append(block)
        return "\n\n---\n\n".join(blocks)

    def _extract_cited_clauses(self, text: str, resolved_clauses: List[ResolvedClause]) -> List[str]:
        """Extract all cited clause IDs and transitional rule references from answer text."""
        cited = set()
        # Find all §x.x.x patterns in text
        for match in re.finditer(r"§\d+\.\d+\.\d+(?:\([a-zA-Z0-9]+\))?", text):
            cited.add(match.group(0))

        # Also include any §5.x rule references
        for match in re.finditer(r"§5\.\d+", text):
            cited.add(match.group(0))

        # Ensure all supplied resolved clauses are captured
        for rc in resolved_clauses:
            if rc.clause_id in text or len(resolved_clauses) == 1:
                cited.add(rc.clause_id)
                if rc.is_amended and rc.transitional_rule_applied:
                    rule_match = re.search(r"§5\.\d+", rc.transitional_rule_applied)
                    if rule_match:
                        cited.add(rule_match.group(0))

        return sorted(list(cited))

    def _offline_fallback_synthesis(
        self,
        query: str,
        resolved_clauses: List[ResolvedClause],
        date_str: Optional[str]
    ) -> str:
        """Deterministic template-based grounded answer synthesis for offline/fallback mode."""
        lines = []
        if date_str:
            lines.append(f"For claim date {date_str}:")
        else:
            lines.append("Based on the company Policy Manual:")

        for rc in resolved_clauses:
            if rc.is_amended:
                lines.append(
                    f"According to {rc.clause_id} (as amended by {rc.applied_source}), "
                    f"{rc.effective_text.strip()}."
                )
                lines.append(f"Applicable transitional provision: {rc.transitional_rule_applied} ({rc.explanation}).")
            else:
                lines.append(
                    f"According to {rc.clause_id} of the base policy manual, "
                    f"{rc.effective_text.strip()}."
                )

        return "\n\n".join(lines)

    def build_answer(
        self,
        query: str,
        resolved_clauses: List[ResolvedClause],
        claim_date: Optional[date] = None
    ) -> GroundedAnswer:
        """Synthesize answer with citations from the provided resolved clauses.

        Args:
            query: User's question.
            resolved_clauses: Date-resolved legally correct clause texts.
            claim_date: Effective date of the claim.

        Returns:
            GroundedAnswer containing answer text, cited clauses, and metadata.
        """
        if not resolved_clauses:
            return GroundedAnswer(
                answer_text="No policy clauses available to answer this inquiry.",
                cited_clauses=[],
                is_refusal=True,
                applied_date_context=claim_date.isoformat() if claim_date else None,
                question=query,
                claim_date=claim_date,
                confidence=0.0
            )

        date_context = claim_date.isoformat() if claim_date else None
        transitional_summaries = [
            f"{rc.clause_id}: {rc.transitional_rule_applied} - {rc.explanation}"
            for rc in resolved_clauses if rc.is_amended
        ]
        transitional_summary_text = "\n".join(transitional_summaries) if transitional_summaries else None

        # Build citations metadata list
        citations: List[Citation] = []
        for rc in resolved_clauses:
            citations.append(
                Citation(
                    clause_id=rc.clause_id,
                    version=rc.applied_source,
                    transitional_rule=rc.transitional_rule_applied if rc.is_amended else None,
                    source_quote=rc.effective_text[:120]
                )
            )

        context_block = self._build_context_block(resolved_clauses)
        answer_text = ""

        # Attempt generation via Gemini LLM if available
        if self.llm is not None:
            prompt = (
                "You are an authoritative policy assistant named 'The Grounded Answer'.\n"
                "Answer the user's question using ONLY the legally resolved policy clauses provided below.\n"
                "CRITICAL RULES:\n"
                "1. Cite the exact clause ID (e.g. §6.4.1(a), §1.1.1) for every policy entitlement, rate, or limit.\n"
                "2. If an amendment applies, cite the governing transitional rule (e.g. §5.1, §5.2) and explain the date rationale.\n"
                "3. Do NOT make up any external facts, figures, or policies.\n\n"
                f"Question: {query}\n"
                f"Effective Claim Date: {date_context or 'Not specified'}\n\n"
                f"--- RESOLVED POLICY CLAUSES ---\n{context_block}\n\n"
                "Grounded Answer:"
            )
            try:
                response = self.llm.invoke(prompt)
                answer_text = response.content if hasattr(response, "content") else str(response)
            except Exception:
                answer_text = ""

        # Use deterministic offline synthesis if LLM is unavailable or failed
        if not answer_text:
            answer_text = self._offline_fallback_synthesis(query, resolved_clauses, date_context)

        cited_clauses = self._extract_cited_clauses(answer_text, resolved_clauses)

        return GroundedAnswer(
            answer_text=answer_text.strip(),
            cited_clauses=cited_clauses,
            is_refusal=False,
            applied_date_context=date_context,
            transitional_summary=transitional_summary_text,
            question=query,
            claim_date=claim_date,
            citations=citations,
            confidence=1.0
        )
