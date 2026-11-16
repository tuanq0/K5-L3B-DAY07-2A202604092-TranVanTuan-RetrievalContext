"""Checks lexical retrieval, RRF fusion, and retrieval-only evaluation."""
import unittest

from starter.benchmark_retrievers import evaluate_retrievers
from starter.retrieval_methods import HybridRRF, PgTfidfRetriever


CHUNKS = [
    {
        "id": "c1",
        "source_id": "doc",
        "source": "doc.pdf",
        "page": 1,
        "text": "Đất chợ dân sinh thuộc đất công cộng.",
        "metadata": {"section": "public"},
    },
    {
        "id": "c2",
        "source_id": "doc",
        "source": "doc.pdf",
        "page": 2,
        "text": "Đất nông nghiệp dùng để trồng lúa.",
        "metadata": {"section": "agriculture"},
    },
]


class FakeRetriever:
    def __init__(self, results):
        self.results = results
        self.calls = []

    def search(self, query, *, filters=None, top_k=3):
        self.calls.append((query, filters, top_k))
        return self.results[:top_k]


class RetrievalMethodTests(unittest.TestCase):
    def test_tfidf_retriever_uses_corpus_and_metadata_filter(self):
        retriever = PgTfidfRetriever(
            "postgresql://local/test",
            dimensions=768,
            chunks=CHUNKS,
        )

        results = retriever.search(
            "đất chợ dân sinh", filters={"section": "public"}, top_k=2
        )

        self.assertEqual([result["id"] for result in results], ["c1"])
        self.assertEqual(results[0]["strategy"], "tfidf")
        self.assertEqual(
            retriever.search(
                "đất chợ dân sinh", filters={"section": "PUBLIC"}, top_k=2
            ),
            [],
        )

    def test_hybrid_rrf_fuses_ranks_and_keeps_component_details(self):
        dense = FakeRetriever([{**CHUNKS[0], "score": 0.8, "strategy": "A"}])
        lexical = FakeRetriever(
            [{**CHUNKS[1], "score": 0.7}, {**CHUNKS[0], "score": 0.6}]
        )
        hybrid = HybridRRF(dense, lexical, candidate_depth=5)

        results = hybrid.search("đất chợ", top_k=2)

        self.assertEqual([result["id"] for result in results], ["c1", "c2"])
        self.assertEqual(results[0]["strategy"], "hybrid-rrf")
        self.assertEqual(results[0]["metadata"]["dense_rank"], 1)
        self.assertEqual(results[0]["metadata"]["lexical_rank"], 2)
        self.assertEqual(dense.calls[0][2], 5)
        self.assertEqual(lexical.calls[0][2], 5)

    def test_evaluator_does_not_send_gold_to_retrievers(self):
        retriever = FakeRetriever([{**CHUNKS[0], "score": 0.8}])
        question = {
            "id": "Q1",
            "split": "dev",
            "question": "Đất chợ dân sinh là gì?",
            "gold": [
                {
                    "source_id": "doc",
                    "page": 1,
                    "evidence": "đất công cộng",
                }
            ],
        }

        traces, summary = evaluate_retrievers(
            {"lexical-tfidf": retriever}, [question]
        )

        self.assertEqual(traces[0]["hitk"], True)
        self.assertEqual(summary["methods"][0]["answerable_n"], 1)
        self.assertEqual(retriever.calls[0][0], question["question"])
        self.assertIsNone(retriever.calls[0][1])

    def test_evaluator_keeps_hit3_fixed_when_top_k_is_larger(self):
        results = [
            {**CHUNKS[1], "id": f"miss-{rank}", "score": 1.0 - rank / 10}
            for rank in range(1, 4)
        ]
        results.append(
            {
                **CHUNKS[0],
                "id": "gold",
                "score": 0.5,
                "text": "Bằng chứng đất chợ dân sinh.",
            }
        )
        retriever = FakeRetriever(results)
        question = {
            "id": "Q3",
            "split": "dev",
            "question": "Đất chợ dân sinh là gì?",
            "gold": [
                {
                    "source_id": "doc",
                    "page": 1,
                    "evidence": "Bằng chứng đất chợ dân sinh",
                }
            ],
        }

        traces, summary = evaluate_retrievers(
            {"lexical-tfidf": retriever}, [question], top_k=4
        )

        self.assertFalse(traces[0]["hit3"])
        self.assertTrue(traces[0]["hitk"])
        self.assertEqual(summary["methods"][0]["hit3"]["hits"], 0)
        self.assertEqual(summary["methods"][0]["hitk"]["hits"], 1)

    def test_evaluator_returns_three_for_fixed_hit3_when_top_k_is_smaller(self):
        results = [
            {**CHUNKS[1], "id": f"miss-{rank}", "score": 1.0 - rank / 10}
            for rank in range(1, 3)
        ]
        results.append(
            {
                **CHUNKS[0],
                "id": "gold",
                "score": 0.5,
                "text": "Bằng chứng đất chợ dân sinh.",
            }
        )
        retriever = FakeRetriever(results)
        question = {
            "id": "Q4",
            "split": "dev",
            "question": "Đất chợ dân sinh là gì?",
            "gold": [
                {
                    "source_id": "doc",
                    "page": 1,
                    "evidence": "Bằng chứng đất chợ dân sinh",
                }
            ],
        }

        traces, summary = evaluate_retrievers(
            {"lexical-tfidf": retriever}, [question], top_k=2
        )

        self.assertEqual(retriever.calls[0][2], 3)
        self.assertTrue(traces[0]["hit3"])
        self.assertFalse(traces[0]["hitk"])
        self.assertEqual(summary["methods"][0]["hit3"]["hits"], 1)
        self.assertEqual(summary["methods"][0]["hitk"]["hits"], 0)

    def test_unanswerable_questions_are_excluded_from_retrieval_metrics(self):
        retriever = FakeRetriever([{**CHUNKS[0], "score": 0.8}])
        question = {
            "id": "Q2",
            "split": "dev",
            "question": "Giá cụ thể của thửa đất là bao nhiêu?",
            "answerable": False,
            "expected_behavior": "search_and_abstain",
            "absence_reason": "Không có hồ sơ thửa đất trong corpus.",
        }

        traces, summary = evaluate_retrievers(
            {"lexical-tfidf": retriever}, [question]
        )

        self.assertIsNone(traces[0]["hit1"])
        self.assertIsNone(traces[0]["results"][0]["hit"])
        metrics = summary["methods"][0]
        self.assertEqual(metrics["answerable_n"], 0)
        self.assertEqual(metrics["unanswerable_n"], 1)
        self.assertEqual(metrics["hitk"]["total"], 0)
        self.assertIsNone(metrics["hitk"]["rate"])


if __name__ == "__main__":
    unittest.main()
