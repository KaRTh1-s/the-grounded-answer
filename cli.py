"""cli.py

Command-Line Interface for 'The Grounded Answer'.
Provides interactive or flag-based execution, dynamic date prompting for
amended clauses, and clean, authoritative output formatting.
"""

import os
import sys
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

import argparse
from datetime import datetime, date
from typing import Optional

from main import GroundedAnswerPipeline
from answer_builder import GroundedAnswer
from refusal_gate import RefusalEvaluation


def parse_args() -> argparse.Namespace:
    """Parse CLI options and positional question arguments."""
    parser = argparse.ArgumentParser(
        prog="grounded-answer",
        description="The Grounded Answer: Date-Aware Policy RAG Assistant"
    )
    parser.add_argument(
        "positional_question",
        type=str,
        nargs="?",
        default=None,
        help="The policy question to ask."
    )
    parser.add_argument(
        "-q", "--query",
        dest="flag_query",
        type=str,
        default=None,
        help="The policy question to ask (alternative to positional argument)."
    )
    parser.add_argument(
        "-d", "--date", "--claim-date",
        dest="claim_date",
        type=str,
        default=None,
        help="Effective claim or expense date (YYYY-MM-DD)."
    )
    parser.add_argument(
        "-e", "--event-date",
        dest="event_date",
        type=str,
        default=None,
        help="Specific change-of-circumstance or assessment period date (YYYY-MM-DD)."
    )
    return parser.parse_args()


def prompt_for_date(prompt_text: str = "Please enter the claim date (YYYY-MM-DD): ") -> date:
    """Interactively prompt user for a valid claim date."""
    while True:
        try:
            raw = input(prompt_text).strip()
            if not raw:
                print("Error: Date cannot be empty.", file=sys.stderr)
                continue
            return datetime.strptime(raw, "%Y-%m-%d").date()
        except ValueError:
            print("Invalid date format. Please use YYYY-MM-DD (e.g. 2026-03-15).", file=sys.stderr)
        except (KeyboardInterrupt, EOFError):
            print("\nOperation cancelled by user.", file=sys.stderr)
            sys.exit(0)


def print_banner() -> None:
    """Print clean CLI banner."""
    print("\n" + "=" * 65)
    print("      THE GROUNDED ANSWER - DATE-AWARE POLICY ASSISTANT")
    print("=" * 65)


def print_result(query: str, result: GroundedAnswer | RefusalEvaluation) -> None:
    """Display clean, structured response output."""
    print("\n[QUESTION]: " + query)
    print("-" * 65)

    if isinstance(result, RefusalEvaluation) or getattr(result, "should_refuse", False) or getattr(result, "is_refusal", False):
        reason = getattr(result, "reason", "OUT_OF_SCOPE") or "REFUSAL"
        message = getattr(result, "message", "") or getattr(result, "answer_text", "")
        contact = getattr(result, "suggested_contact", "HR Policy Desk")

        print(f"\n[STATUS]: REFUSED / GUARDED")
        print(f"[REASON]: {reason}")
        print(f"\n[EXPLANATION]:\n{message}")
        print(f"\n[SUGGESTED CONTACT]: {contact}")
    else:
        print(f"\n[STATUS]: GROUNDED & VERIFIED")
        if result.applied_date_context:
            print(f"[CLAIM DATE]: {result.applied_date_context}")
        if result.cited_clauses:
            print(f"[CITED CLAUSES]: {', '.join(result.cited_clauses)}")
        if result.transitional_summary:
            print(f"\n[TRANSITIONAL RATIONALE]:\n{result.transitional_summary}")

        print(f"\n[ANSWER]:\n{result.answer_text}")

    print("=" * 65 + "\n")


def main() -> None:
    """CLI main entry point."""
    args = parse_args()
    query = args.flag_query or args.positional_question

    print_banner()

    if not query:
        try:
            query = input("Enter your policy question: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nOperation cancelled by user.", file=sys.stderr)
            sys.exit(0)

    if not query:
        print("Error: Question cannot be empty.", file=sys.stderr)
        sys.exit(1)

    pipeline = GroundedAnswerPipeline()

    claim_date_val: Optional[date] = None
    if args.claim_date:
        try:
            claim_date_val = datetime.strptime(args.claim_date.strip(), "%Y-%m-%d").date()
        except ValueError:
            print(f"Error: Invalid date format '{args.claim_date}'. Use YYYY-MM-DD.", file=sys.stderr)
            sys.exit(1)
    else:
        # Check if the query touches any date-sensitive amended clauses
        if pipeline.requires_date_context(query):
            print("\n[!] The retrieved policy provisions contain date-dependent amendment rules.")
            claim_date_val = prompt_for_date("Please enter the claim date (YYYY-MM-DD): ")

    result = pipeline.run_query(
        query=query,
        claim_date=claim_date_val,
        event_date=args.event_date
    )

    print_result(query, result)


if __name__ == "__main__":
    main()
