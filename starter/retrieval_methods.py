"""Lexical TF-IDF and hybrid RRF retrievers over the Day 6 pgvector corpus."""
import math
from typing import Any, Protocol

import psycopg

from starter.baseline import TfidfIndex


class SearchRetriever(Protocol):
    def search(
        self,
        query: str,
        *,
        filters: dict[str, Any] | None = None,
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        ...


class PgTfidfRetriever:
    """Run the existing TF-IDF baseline over chunks in the Day 6 database."""

    def __init__(
        self,
        database_url: str,
        *,
        dimensions: int,
        strategy: str = "A",
        chunks: list[dict[str, Any]] | None = None,
    ) -> None:
        if not isinstance(database_url, str) or not database_url.strip():
            raise ValueError("database_url must not be empty")
        if isinstance(dimensions, bool) or not isinstance(dimensions, int) or dimensions <= 0:
            raise ValueError("dimensions must be a positive integer")
        if not isinstance(strategy, str) or not strategy.strip():
            raise ValueError("strategy must not be empty")

        self.database_url = database_url
        self.dimensions = dimensions
        self.strategy = strategy.strip()
        self.table = f"rag_chunks_{dimensions}"
        self.chunks = chunks if chunks is not None else self._load_chunks()
        if not self.chunks:
            raise RuntimeError(
                f"No chunks found for strategy {self.strategy!r} in {self.table}."
            )
        self.index = TfidfIndex(self.chunks)

    def _load_chunks(self) -> list[dict[str, Any]]:
        with psycopg.connect(self.database_url) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    SELECT chunk_id, document_id, source, page, text, metadata
                    FROM {self.table}
                    WHERE strategy = %s
                    ORDER BY chunk_id
                    """,
                    (self.strategy,),
                )
                rows = cursor.fetchall()

        chunks = []
        for chunk_id, document_id, source, page, text, metadata in rows:
            if not isinstance(metadata, dict):
                raise TypeError(f"metadata for chunk {chunk_id!r} must be an object")
            chunks.append(
                {
                    "id": chunk_id,
                    "source_id": document_id,
                    "source": source,
                    "page": page,
                    "text": text,
                    "metadata": metadata,
                }
            )
        return chunks

    @staticmethod
    def _validated_filters(filters: dict[str, Any] | None) -> dict[str, str]:
        if filters is None:
            return {}
        if not isinstance(filters, dict):
            raise TypeError("filters must be a dictionary or None")

        validated = {}
        for key, value in filters.items():
            if not isinstance(key, str) or not key.strip():
                raise ValueError("metadata filter keys must be non-empty strings")
            if not isinstance(value, (str, int, float, bool)) or value is None:
                raise ValueError(
                    f"metadata filter {key!r} must have a non-null scalar value"
                )
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"metadata filter {key!r} must be finite")
            validated[key] = (
                "true" if value else "false"
            ) if isinstance(value, bool) else str(value)
        return validated

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
        selected_filters = self._validated_filters(filters)

        ranked = self.index.search(query, top_k=len(self.chunks))
        results = []
        for result in ranked:
            metadata = result["metadata"]
            matches = True
            for key, value in selected_filters.items():
                actual = metadata.get(key)
                if isinstance(actual, bool):
                    actual_value = "true" if actual else "false"
                elif actual is None:
                    actual_value = ""
                else:
                    actual_value = str(actual)
                if actual_value != value:
                    matches = False
                    break
            if matches:
                results.append({**result, "strategy": "tfidf"})
                if len(results) == top_k:
                    break
        return results


class HybridRRF:
    """Fuse dense and lexical rankings with Reciprocal Rank Fusion."""

    def __init__(
        self,
        dense: SearchRetriever,
        lexical: SearchRetriever,
        *,
        rrf_k: int = 60,
        candidate_depth: int = 20,
    ) -> None:
        if isinstance(rrf_k, bool) or not isinstance(rrf_k, int) or rrf_k <= 0:
            raise ValueError("rrf_k must be a positive integer")
        if (
            isinstance(candidate_depth, bool)
            or not isinstance(candidate_depth, int)
            or candidate_depth <= 0
        ):
            raise ValueError("candidate_depth must be a positive integer")
        self.retrievers = (("dense", dense), ("lexical", lexical))
        self.rrf_k = rrf_k
        self.candidate_depth = candidate_depth

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

        candidate_k = max(top_k, self.candidate_depth)
        fused_scores: dict[str, float] = {}
        chunks: dict[str, dict[str, Any]] = {}
        ranks: dict[str, dict[str, int]] = {"dense": {}, "lexical": {}}
        component_scores: dict[str, dict[str, float]] = {
            "dense": {},
            "lexical": {},
        }

        for name, retriever in self.retrievers:
            results = retriever.search(
                query, filters=filters, top_k=candidate_k
            )
            for rank, result in enumerate(results, start=1):
                chunk_id = result.get("id")
                if not isinstance(chunk_id, str) or not chunk_id:
                    raise ValueError(f"{name} retriever returned a chunk without an ID")
                if chunk_id in ranks[name]:
                    continue
                score = result.get("score")
                if isinstance(score, bool) or not isinstance(score, (int, float)):
                    raise ValueError(f"{name} retriever returned an invalid score")
                if not math.isfinite(float(score)):
                    raise ValueError(f"{name} retriever returned a non-finite score")

                ranks[name][chunk_id] = rank
                component_scores[name][chunk_id] = float(score)
                fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + (
                    1.0 / (self.rrf_k + rank)
                )
                chunks.setdefault(chunk_id, result)

        ordered_ids = sorted(
            fused_scores,
            key=lambda chunk_id: (-fused_scores[chunk_id], chunk_id),
        )
        fused = []
        for chunk_id in ordered_ids[:top_k]:
            result = chunks[chunk_id]
            metadata = result.get("metadata")
            if not isinstance(metadata, dict):
                raise ValueError(f"metadata for chunk {chunk_id!r} must be an object")
            details = dict(metadata)
            for name in ("dense", "lexical"):
                if chunk_id in ranks[name]:
                    details[f"{name}_rank"] = ranks[name][chunk_id]
                    details[f"{name}_score"] = component_scores[name][chunk_id]
            fused.append(
                {
                    **result,
                    "metadata": details,
                    "score": fused_scores[chunk_id],
                    "strategy": "hybrid-rrf",
                }
            )
        return fused


def create_retrievers() -> dict[str, SearchRetriever]:
    """Build dense, lexical, and hybrid retrievers over the same Day 6 table."""
    from starter.pgvector_retriever import PgVectorRetriever

    dense = PgVectorRetriever()
    lexical = PgTfidfRetriever(
        dense.database_url,
        dimensions=dense.embedder.dimensions,
        strategy=dense.strategy,
    )
    return {
        "dense": dense,
        "lexical-tfidf": lexical,
        "hybrid-rrf": HybridRRF(dense, lexical),
    }
