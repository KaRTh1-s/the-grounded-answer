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

    @staticmethod
    def _clean_clause_text(text: str, clause_id: str) -> str:
        """Clean raw markdown formatting, bold tags, section numerals, and amendment syntax."""
        cleaned = text.strip()

        # 1. Strip bold amendment paragraph numbering like **1.1**, **2.1**, **3.1** FIRST
        cleaned = re.sub(r"^\*\*\d+\.\d+\*\*\s*", "", cleaned)
        # Also strip bare amendment numerals at start (e.g. "2.1 ", "1.1 ")
        cleaned = re.sub(r"^\d+\.\d+\s+", "", cleaned)

        # 2. Strip leading markdown bullets, headers, dashes, or remaining stars
        cleaned = re.sub(r"^[-*#]+\s*", "", cleaned)

        # 3. Strip any remaining bold/italic markdown markers
        cleaned = cleaned.replace("**", "").replace("__", "")

        # 4. Clean redundant clause prefixes:
        #    - "In §6.4.1(a), "
        #    - "§6.4.2 " (from base manual chunks like "§6.4.2 In calculating...")
        cleaned = re.sub(rf"^In\s+{re.escape(clause_id)},?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"^In\s+§\d+\.\d+\.\d+(?:\([a-zA-Z0-9]+\))?,?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(rf"^{re.escape(clause_id)}\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"^§\d+\.\d+\.\d+(?:\([a-zA-Z0-9]+\))?\s*", "", cleaned, flags=re.IGNORECASE)

        # 5. Ensure capitalization, single period at end, and clean spacing
        cleaned = cleaned.strip()
        if cleaned:
            cleaned = cleaned[0].upper() + cleaned[1:]
            if not cleaned.endswith("."):
                cleaned += "."
            cleaned = re.sub(r"\.{2,}$", ".", cleaned)

        return cleaned

    def _offline_fallback_synthesis(
        self,
        query: str,
        resolved_clauses: List[ResolvedClause],
        date_str: Optional[str]
    ) -> str:
        """Deterministic template-based grounded answer synthesis for offline/fallback mode."""
        if not resolved_clauses:
            return "No policy clauses available to answer this inquiry."

        paragraphs = []
        header = f"As of claim date {date_str}:" if date_str else "Under the company policy:"
        paragraphs.append(header)

        for rc in resolved_clauses:
            text = self._clean_clause_text(rc.effective_text, rc.clause_id)

            if rc.is_amended:
                paragraphs.append(
                    f"Under {rc.clause_id} (as updated by {rc.applied_source}), {text} "
                    f"This rate is legally binding pursuant to transitional rule {rc.transitional_rule_applied}."
                )
            else:
                paragraphs.append(f"Under {rc.clause_id} of the base policy manual, {text}")

        return "\n\n".join(paragraphs)

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
                "Your goal is to synthesize the provided legally effective policy clauses into a cohesive, "
                "professional, and natural response that directly answers the user's question.\n\n"
                "CRITICAL GROUNDING INSTRUCTIONS:\n"
                "1. Cohesive Synthesis: Synthesize the relevant policy rules into a clean, cohesive, and natural "
                "response that directly answers the user's question. Do NOT list raw verbatim quotes line-by-line.\n"
                "2. Explicit Inline Citations: Every entitlement, rate, dollar figure, deadline, or policy requirement "
                "MUST be immediately followed by its exact clause citation in parentheses or inline (e.g. §6.4.1(a), §1.1.1).\n"
                "3. Active Claim Date Context: Explicitly contextualize the effective claim date in your explanation. "
                "If an amendment modified the rate or rule for that date, cite the governing transitional rule (e.g. §5.1, §5.2) "
                "and explain the legal rationale.\n"
                "4. Strict Relevance Filter: Mention and cite ONLY clauses from the context that directly answer the core question. "
                "Do NOT cite or discuss irrelevant clauses from the context block.\n"
                "5. Zero Hallucinations: Strictly rely on the provided context. Do NOT introduce external policies, numbers, "
                "or ungrounded assumptions.\n\n"
                f"User Question: {query}\n"
                f"Active Claim Date: {date_context or 'Not specified'}\n\n"
                f"--- RESOLVED POLICY CLAUSES (AUTHORIZED CONTEXT) ---\n{context_block}\n\n"
                "Grounded Response:"
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
