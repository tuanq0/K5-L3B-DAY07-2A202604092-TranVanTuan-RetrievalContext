"""Compare dense, lexical TF-IDF, and hybrid RRF on one benchmark split."""
import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

from starter.baseline import hit, pack_context
from starter.retrieval_methods import create_retrievers


def _read_questions(path: Path, split: str) -> list[dict[str, Any]]:
    questions = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(questions, list):
        raise ValueError("questions must be a JSON array")
    selected = [question for question in questions if question.get("split") == split]
    if not selected:
        raise ValueError(f"No questions found for split {split!r}")

    ids = set()
    for question in selected:
        query_id = question.get("id")
        if not isinstance(query_id, str) or not query_id.strip() or query_id in ids:
            raise ValueError("question IDs must be non-empty and unique")
        ids.add(query_id)
        if not isinstance(question.get("question"), str) or not question["question"].strip():
            raise ValueError(f"question {query_id!r} must contain question text")
        gold = question.get("gold", [])
        if not isinstance(gold, list):
            raise ValueError(f"question {query_id!r} gold must be a list")
        answerable = question.get("answerable", bool(gold))
        if not isinstance(answerable, bool) or answerable != bool(gold):
            raise ValueError(
                f"question {query_id!r} answerable label must agree with gold"
            )
        if not answerable and (
            question.get("expected_behavior") != "search_and_abstain"
            or not isinstance(question.get("absence_reason"), str)
            or not question["absence_reason"].strip()
        ):
            raise ValueError(
                f"unanswerable question {query_id!r} needs abstention behavior and absence_reason"
            )
    return selected


def evaluate_retrievers(
    retrievers: dict[str, Any],
    questions: list[dict[str, Any]],
    *,
    top_k: int = 3,
    budget: int = 800,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not retrievers:
        raise ValueError("retrievers must not be empty")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    if isinstance(budget, bool) or not isinstance(budget, int) or budget <= 0:
        raise ValueError("budget must be a positive integer")

    traces = []
    for question in questions:
        gold = question.get("gold", [])
        answerable = bool(gold)
        filters = question.get("filter")
        for name, retriever in retrievers.items():
            started = time.perf_counter()
            ranking_depth = max(top_k, 3)
            results = retriever.search(
                question["question"], filters=filters, top_k=ranking_depth
            )
            elapsed_ms = (time.perf_counter() - started) * 1000
            context_results = results[:top_k]
            context, context_ids = pack_context(context_results, budget=budget)
            marked = [
                {
                    **result,
                    "hit": hit(result, gold) if answerable else None,
                }
                for result in results
            ]
            hit1 = bool(marked and marked[0]["hit"]) if answerable else None
            hit3 = any(result["hit"] for result in marked[:3]) if answerable else None
            hitk = any(result["hit"] for result in marked[:top_k]) if answerable else None
            context_hit = (
                any(
                    result["hit"] and result["id"] in context_ids
                    for result in marked[:top_k]
                )
                if answerable
                else None
            )
            traces.append(
                {
                    "query_id": question["id"],
                    "split": question["split"],
                    "question": question["question"],
                    "strategy": name,
                    "top_k": top_k,
                    "answerable": answerable,
                    "gold": gold,
                    "results": marked,
                    "context": context,
                    "context_ids": context_ids,
                    "context_chars": len(context),
                    "elapsed_ms": elapsed_ms,
                    "hit1": hit1,
                    "hit3": hit3,
                    "hitk": hitk,
                    "context_hit": context_hit,
                }
            )

    summaries = []
    for name in retrievers:
        method_traces = [trace for trace in traces if trace["strategy"] == name]
        answerable_traces = [
            trace for trace in method_traces if trace["answerable"]
        ]

        def metric(name: str) -> dict[str, Any]:
            hits = sum(trace[name] is True for trace in answerable_traces)
            total = len(answerable_traces)
            return {
                "hits": hits,
                "total": total,
                "rate": hits / total if total else None,
            }

        summaries.append(
            {
                "strategy": name,
                "n": len(method_traces),
                "answerable_n": len(answerable_traces),
                "unanswerable_n": len(method_traces) - len(answerable_traces),
                "hit1": metric("hit1"),
                "hit3": metric("hit3"),
                "hitk": metric("hitk"),
                "context_hit": metric("context_hit"),
                "avg_context_chars": (
                    statistics.mean(
                        trace["context_chars"] for trace in method_traces
                    )
                    if method_traces
                    else 0
                ),
                "median_ms": (
                    statistics.median(
                        trace["elapsed_ms"] for trace in method_traces
                    )
                    if method_traces
                    else 0
                ),
            }
        )
    return traces, {"methods": summaries}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark dense, lexical TF-IDF, and hybrid RRF retrieval."
    )
    parser.add_argument("--questions", required=True, type=Path)
    parser.add_argument("--split", required=True, choices=("dev", "test"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--budget", type=int, default=800)
    args = parser.parse_args()

    questions = _read_questions(args.questions, args.split)
    retrievers = create_retrievers()
    traces, summary = evaluate_retrievers(
        retrievers, questions, top_k=args.top_k, budget=args.budget
    )
    summary.update(
        {
            "split": args.split,
            "n": len(questions),
            "top_k": args.top_k,
            "budget_chars": args.budget,
        }
    )

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "traces.json").write_text(
        json.dumps(traces, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
