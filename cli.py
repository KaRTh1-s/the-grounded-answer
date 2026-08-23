"""cli.py

Command-line entry point for 'The Grounded Answer'.
One question in, one grounded answer out.
Accepts an optional --date flag or interactively prompts for the claim date if omitted.
"""

import argparse
from datetime import datetime, date
import sys


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="grounded-answer",
        description="CLI RAG assistant for date-aware, cited policy manual inquiries."
    )
    parser.add_argument(
        "question",
        type=str,
        nargs="?",
        help="The policy question to ask."
    )
    parser.add_argument(
        "-d", "--date",
        dest="claim_date",
        type=str,
        default=None,
        help="The claim/expense date (YYYY-MM-DD). If omitted, you will be prompted."
    )
    return parser.parse_args()


def prompt_for_date() -> date:
    """Prompt the user interactively for a valid claim date."""
    while True:
        raw_input = input("Enter claim date (YYYY-MM-DD): ").strip()
        try:
            return datetime.strptime(raw_input, "%Y-%m-%d").date()
        except ValueError:
            print("Invalid date format. Please use YYYY-MM-DD (e.g., 2026-03-15).", file=sys.stderr)


def main() -> None:
    """CLI execution entrypoint."""
    args = parse_args()

    if not args.question:
        args.question = input("Enter policy question: ").strip()
        if not args.question:
            print("Error: Question cannot be empty.", file=sys.stderr)
            sys.exit(1)

    if args.claim_date:
        try:
            claim_date = datetime.strptime(args.claim_date, "%Y-%m-%d").date()
        except ValueError:
            print(f"Error: Invalid date '{args.claim_date}'. Use YYYY-MM-DD.", file=sys.stderr)
            sys.exit(1)
    else:
        claim_date = prompt_for_date()

    print(f"\n[Evaluating Question]: {args.question}")
    print(f"[Claim Date]: {claim_date.isoformat()}")
    print("\n(Stub) Pipeline execution will produce grounded answer or refusal in the next phase.")


if __name__ == "__main__":
    main()
