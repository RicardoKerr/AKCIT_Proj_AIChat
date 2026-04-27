import json
from typing import List

import psycopg


class PostgresVectorStore:
    def __init__(self, dsn: str):
        self.dsn = dsn

    def _connect(self):
        return psycopg.connect(self.dsn)

    @staticmethod
    def _to_vector_literal(values: List[float]) -> str:
        return "[" + ",".join(f"{float(v):.8f}" for v in values) + "]"

    def ensure_schema(self) -> None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS rag_chunks (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        content TEXT NOT NULL,
                        embedding VECTOR(1536) NOT NULL,
                        metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
                        created_at TIMESTAMP NOT NULL DEFAULT NOW()
                    );
                    """
                )
            conn.commit()

    def upsert_chunks(self, chunks: List[dict], embeddings: List[List[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("Quantidade de chunks e embeddings deve ser igual")

        with self._connect() as conn:
            with conn.cursor() as cur:
                for chunk, embedding in zip(chunks, embeddings):
                    vector_literal = self._to_vector_literal(embedding)
                    metadata_json = json.dumps(chunk.get("metadata", {}), ensure_ascii=True)
                    cur.execute(
                        """
                        INSERT INTO rag_chunks (content, embedding, metadata)
                        VALUES (%s, %s::vector, %s::jsonb);
                        """,
                        (chunk["content"], vector_literal, metadata_json),
                    )
            conn.commit()

    def search_similar(
        self,
        query_embedding: List[float],
        top_k: int = 4,
        similarity_threshold: float = 0.6,
    ) -> List[dict]:
        vector_literal = self._to_vector_literal(query_embedding)
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                        content,
                        metadata,
                        1 - (embedding <=> %s::vector) AS similarity
                    FROM rag_chunks
                    WHERE 1 - (embedding <=> %s::vector) >= %s
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s;
                    """,
                    (vector_literal, vector_literal, similarity_threshold, vector_literal, top_k),
                )
                rows = cur.fetchall()

        results: List[dict] = []
        for row in rows:
            results.append(
                {
                    "content": row[0],
                    "metadata": row[1] or {},
                    "similarity": row[2],
                }
            )
        return results
