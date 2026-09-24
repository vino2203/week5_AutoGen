"""Command-line entry point: evaluate a project input file and print the recommendation.

    python run_manager.py samples/whatsapp_gmail_faq.json
    python run_manager.py samples/whatsapp_gmail_faq.json --json

Needs OPENAI_API_KEY in Backend/.env.
"""
import argparse
import asyncio
import sys

from agents.config import load_config
from agents.human_loop_agent import _message
from agents.llm import LLMNotConfigured, get_llm
from agents.manager import Manager
from agents.schemas import ProjectInput


async def main(argv: list[str] | None = None, llm=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_file", help="JSON file with the project input (see samples/)")
    parser.add_argument("--json", action="store_true", help="print the full session as JSON")
    args = parser.parse_args(argv)

    with open(args.input_file, encoding="utf-8") as f:
        project = ProjectInput.model_validate_json(f.read())
    if llm is None:
        try:
            llm = get_llm()
        except LLMNotConfigured as e:
            print(e, file=sys.stderr)
            return 2
    session = await Manager(llm, load_config()).start(project)
    print(session.model_dump_json(indent=2) if args.json else _message(session))
    return 0 if session.status == "awaiting_approval" else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
