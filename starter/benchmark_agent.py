"""Run Claude-backed agent evaluations over one benchmark split."""
import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

from starter.agentic import run_agent
from starter.benchmark_retrievers import _read_questions
from starter.claude_controller import ClaudeController
from starter.retrieval_methods import create_retrievers


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the Claude agent on one benchmark split."
    )
    parser.add_argument("--questions", required=True, type=Path)
    parser.add_argument("--split", required=True, choices=("dev", "test"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--retrieval",
        choices=("dense", "lexical-tfidf", "hybrid-rrf"),
        default="hybrid-rrf",
    )
    parser.add_argument("--max-searches", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--budget", type=int, default=800)
    args = parser.parse_args()

    questions = _read_questions(args.questions, args.split)
    controller = ClaudeController()
    retriever = create_retrievers()[args.retrieval]
    args.output.mkdir(parents=True, exist_ok=True)
    traces_path = args.output / "traces.json"
    traces: list[dict[str, Any]] = []
    _write_json(traces_path, traces)

    for question in questions:
        usage_start = len(controller.usage_records)
        started = time.perf_counter()
        result = run_agent(
            question["question"],
            controller,
            retriever,
            filters=question.get("filter"),
            max_searches=args.max_searches,
            top_k=args.top_k,
            budget=args.budget,
        )
        usage = controller.usage_records[usage_start:]
        events = result["events"]
        traces.append(
            {
                "query_id": question["id"],
                "split": args.split,
                "question": question["question"],
                "answerable": question.get(
                    "answerable", bool(question.get("gold"))
                ),
                "status": result["status"],
                "answer": result["answer"],
                "citations": result["citations"],
                "events": events,
                "search_calls": sum(event["event"] == "search" for event in events),
                "api_calls": len(usage),
                "input_tokens": sum(item["input_tokens"] for item in usage),
                "output_tokens": sum(item["output_tokens"] for item in usage),
                "elapsed_ms": (time.perf_counter() - started) * 1000,
            }
        )
        _write_json(traces_path, traces)

    status_counts = {
        status: sum(trace["status"] == status for trace in traces)
        for status in ("answered", "abstain")
    }
    summary = {
        "split": args.split,
        "n": len(traces),
        "model": controller.model,
        "retrieval": args.retrieval,
        "max_searches": args.max_searches,
        "top_k": args.top_k,
        "budget_chars": args.budget,
        "status_counts": status_counts,
        "queries_with_search": sum(trace["search_calls"] > 0 for trace in traces),
        "search_calls": sum(trace["search_calls"] for trace in traces),
        "api_calls": sum(trace["api_calls"] for trace in traces),
        "input_tokens": sum(trace["input_tokens"] for trace in traces),
        "output_tokens": sum(trace["output_tokens"] for trace in traces),
        "median_ms": statistics.median(trace["elapsed_ms"] for trace in traces),
    }
    _write_json(args.output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
