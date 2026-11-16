"""Unit tests for the PostgreSQL retriever adapter."""
import unittest
from unittest.mock import MagicMock, patch

from starter.pgvector_retriever import PgVectorRetriever


class FakeEmbedder:
    dimensions = 2

    def embed_query(self, query):
        return [0.25, 0.75]


class PgVectorRetrieverTests(unittest.TestCase):
    @patch("starter.pgvector_retriever.psycopg.connect")
    def test_search_maps_day6_rows_and_parameterizes_metadata_filters(self, connect):
        connection = MagicMock()
        cursor = MagicMock()
        connection.__enter__.return_value = connection
        connection.cursor.return_value.__enter__.return_value = cursor
        cursor.fetchall.return_value = [
            ("chunk-1", "doc-1", "source.pdf", 4, "evidence", {"section": "one"}, 0.9)
        ]
        connect.return_value = connection

        retriever = PgVectorRetriever(
            "postgresql://local/test",
            strategy="A",
            embedder=FakeEmbedder(),
        )
        results = retriever.search(
            "query", filters={"section": "one'; DROP TABLE chunks; --"}, top_k=2
        )

        self.assertEqual(
            results,
            [
                {
                    "id": "chunk-1",
                    "source_id": "doc-1",
                    "source": "source.pdf",
                    "page": 4,
                    "text": "evidence",
                    "metadata": {
                        "section": "one",
                        "document_id": "doc-1",
                        "source": "source.pdf",
                        "page": 4,
                    },
                    "score": 0.9,
                    "strategy": "A",
                }
            ],
        )
        sql, parameters = cursor.execute.call_args.args
        self.assertIn("FROM rag_chunks_2", sql)
        self.assertIn("metadata ->> %s = %s", sql)
        self.assertEqual(
            parameters,
            ["[0.25,0.75]", "A", "section", "one'; DROP TABLE chunks; --",
             "[0.25,0.75]", 2],
        )

    def test_rejects_invalid_top_k(self):
        retriever = PgVectorRetriever(
            "postgresql://local/test", embedder=FakeEmbedder()
        )
        with self.assertRaises(ValueError):
            retriever.search("query", top_k=0)

    @patch("starter.pgvector_retriever.load_dotenv")
    def test_requires_embedding_model_configuration(self, load_dotenv):
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(ValueError, "EMBEDDING_MODEL"):
                PgVectorRetriever("postgresql://local/test")


if __name__ == "__main__":
    unittest.main()
