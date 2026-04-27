from typing import List, Optional
from uuid import uuid4

import chromadb
from chromadb.config import Settings


class ChromaVectorStore:
    def __init__(self, collection_name: str = "rag_chunks", persist_dir: str = ".chroma"):
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def reset_collection(self) -> None:
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            # Se a coleção não existir, apenas recria.
            pass

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert_chunks(self, chunks: List[dict], embeddings: List[List[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("Quantidade de chunks e embeddings deve ser igual")

        ids: List[str] = []
        documents: List[str] = []
        metadatas: List[dict] = []

        for chunk in chunks:
            ids.append(str(uuid4()))
            documents.append(chunk["content"])
            metadatas.append(chunk.get("metadata", {}))

        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def search_similar(
        self,
        query_embedding: List[float],
        top_k: int = 4,
        similarity_threshold: float = 0.6,
        source_filter: Optional[List[str]] = None,
    ) -> List[dict]:
        query_args = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }

        if source_filter:
            unique_sources = sorted(set(source_filter))
            query_args["where"] = {"source": {"$in": unique_sources}}

        query_result = self.collection.query(**query_args)

        documents = query_result.get("documents", [[]])[0]
        metadatas = query_result.get("metadatas", [[]])[0]
        distances = query_result.get("distances", [[]])[0]

        results: List[dict] = []
        for document, metadata, distance in zip(documents, metadatas, distances):
            similarity = 1.0 - float(distance)
            if similarity < similarity_threshold:
                continue
            results.append(
                {
                    "content": document,
                    "metadata": metadata or {},
                    "similarity": similarity,
                }
            )

        return results
