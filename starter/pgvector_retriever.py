"""Dense retrieval from the Day 6 PostgreSQL + pgvector store."""
import math
import os
from pathlib import Path
from typing import Any, Protocol

import psycopg
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer


class QueryEmbedder(Protocol):
    dimensions: int

    def embed_query(self, query: str) -> list[float]:
        ...


class SentenceTransformerQueryEmbedder:
    """Create query vectors with the same model and prompt as the Day 6 index."""

    def __init__(self, model_name: str) -> None:
        self._model = SentenceTransformer(model_name, device="cpu")
        dimensions = self._model.get_embedding_dimension()
        if dimensions is None or dimensions <= 0:
            raise RuntimeError(
                f"Could not determine embedding dimensions for {model_name!r}."
            )
        self.dimensions = dimensions

    def embed_query(self, query: str) -> list[float]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must not be empty or whitespace-only")

        options: dict[str, str] = {}
        if "SearchQuery" in self._model.prompts:
            options["prompt_name"] = "SearchQuery"
        encoded = self._model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
            **options,
        )
        if len(encoded) != 1:
            raise RuntimeError(
                f"Model returned {len(encoded)} vectors for one query."
            )
        vector = [float(value) for value in encoded[0]]
        if len(vector) != self.dimensions:
            raise RuntimeError(
                f"Model returned {len(vector)} dimensions; expected {self.dimensions}."
            )
        return vector


class PgVectorRetriever:
    """Retriever compatible with ``agentic.Retriever`` for the Day 6 schema."""

    def __init__(
        self,
        database_url: str | None = None,
        *,
        strategy: str | None = None,
        embedding_model: str | None = None,
        embedder: QueryEmbedder | None = None,
    ) -> None:
        load_dotenv(Path(__file__).resolve().with_name(".env"))

        selected_url = (
            database_url
            if database_url is not None
            else os.getenv("DATABASE_URL", "")
        ).strip()
        if not selected_url:
            raise ValueError(
                "DATABASE_URL is missing. Configure it in starter/.env."
            )
        selected_strategy = (
            strategy if strategy is not None else os.getenv("VECTOR_STRATEGY", "A")
        ).strip()
        if not selected_strategy:
            raise ValueError("VECTOR_STRATEGY must not be empty")

        self.database_url = selected_url
        self.strategy = selected_strategy
        if embedder is None:
            selected_model = embedding_model
            if selected_model is None:
                selected_model = (
                    os.getenv("EMBEDDING_MODEL", "").strip()
                    or os.getenv("SENTENCE_TRANSFORMER_MODEL", "").strip()
                )
            if not selected_model.strip():
                raise ValueError(
                    "EMBEDDING_MODEL is missing. Configure it in starter/.env."
                )
            embedder = SentenceTransformerQueryEmbedder(selected_model)
        if (
            isinstance(embedder.dimensions, bool)
            or not isinstance(embedder.dimensions, int)
            or embedder.dimensions <= 0
        ):
            raise ValueError("embedder dimensions must be a positive integer")
        self.embedder = embedder
        self.table = f"rag_chunks_{embedder.dimensions}"

    @staticmethod
    def _vector_literal(vector: list[float], dimensions: int) -> str:
        if len(vector) != dimensions:
            raise ValueError(
                f"query vector has {len(vector)} dimensions; expected {dimensions}"
            )
        values = []
        for value in vector:
            if isinstance(value, bool):
                raise TypeError("vector values must be numeric, not booleans")
            try:
                number = float(value)
            except (TypeError, ValueError) as error:
                raise TypeError("vector values must be numeric") from error
            if not math.isfinite(number):
                raise ValueError("vector values must be finite numbers")
            values.append(format(number, ".17g"))
        if not any(float(value) != 0.0 for value in values):
            raise ValueError("zero vectors cannot be used with cosine distance")
        return "[" + ",".join(values) + "]"

    def search(
        self,
        query: str,
        *,
        filters: dict[str, Any] | None = None,
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must not be empty or whitespace-only")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        if filters is not None and not isinstance(filters, dict):
            raise TypeError("filters must be a dictionary or None")

        vector = self.embedder.embed_query(query)
        vector_literal = self._vector_literal(vector, self.embedder.dimensions)

        filter_clauses = []
        parameters: list[Any] = [vector_literal, self.strategy]
        for key, value in (filters or {}).items():
            if not isinstance(key, str) or not key.strip():
                raise ValueError("metadata filter keys must be non-empty strings")
            if not isinstance(value, (str, int, float, bool)) or value is None:
                raise ValueError(
                    f"metadata filter {key!r} must have a non-null scalar value"
                )
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"metadata filter {key!r} must be finite")
            if isinstance(value, bool):
                filter_value = "true" if value else "false"
            else:
                filter_value = str(value)
            filter_clauses.append("metadata ->> %s = %s")
            parameters.extend((key, filter_value))

        where_filters = (
            " AND " + " AND ".join(filter_clauses) if filter_clauses else ""
        )
        parameters.extend((vector_literal, top_k))
        sql = f"""
            SELECT chunk_id, document_id, source, page, text, metadata,
                   1 - (embedding <=> %s::vector) AS score
            FROM {self.table}
            WHERE strategy = %s{where_filters}
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """

        with psycopg.connect(self.database_url) as connection:
            with connection.cursor() as cursor:
                cursor.execute(sql, parameters)
                rows = cursor.fetchall()

        results = []
        for chunk_id, document_id, source, page, text, metadata, score in rows:
            if not isinstance(metadata, dict):
                raise TypeError(f"metadata for chunk {chunk_id!r} must be an object")
            result_metadata = {
                **metadata,
                "document_id": document_id,
                "source": source,
                "page": page,
            }
            results.append(
                {
                    "id": chunk_id,
                    "source_id": document_id,
                    "source": source,
                    "page": page,
                    "text": text,
                    "metadata": result_metadata,
                    "score": float(score),
                    "strategy": self.strategy,
                }
            )
        return results
