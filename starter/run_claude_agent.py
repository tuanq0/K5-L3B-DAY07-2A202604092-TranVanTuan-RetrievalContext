"""Run the retrieval agent using Claude and a selected retrieval method."""
import argparse
import json

from starter.agentic import run_agent
from starter.claude_controller import ClaudeController
from starter.retrieval_methods import create_retrievers


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ask a question using Claude and the Day 6 retrieval store."
    )
    parser.add_argument("question")
    parser.add_argument(
        "--retrieval",
        choices=("dense", "lexical-tfidf", "hybrid-rrf"),
        default="hybrid-rrf",
    )
    parser.add_argument("--max-searches", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--budget", type=int, default=800)
    args = parser.parse_args()

    controller = ClaudeController()
    retriever = create_retrievers()[args.retrieval]
    result = run_agent(
        args.question,
        controller,
        retriever,
        max_searches=args.max_searches,
        top_k=args.top_k,
        budget=args.budget,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
