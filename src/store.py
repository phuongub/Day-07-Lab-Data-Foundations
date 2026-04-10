from __future__ import annotations

from typing import Any, Callable

from .chunking import _dot, compute_similarity
from .embeddings import _mock_embed
from .models import Document


class EmbeddingStore:
    """
    A vector store for text chunks.

    Tries to use ChromaDB if available; falls back to an in-memory store.
    The embedding_fn parameter allows injection of mock embeddings for tests.
    """

    def __init__(
        self,
        collection_name: str = "documents",
        embedding_fn: Callable[[str], list[float]] | None = None,
    ) -> None:
        self._embedding_fn = embedding_fn or _mock_embed
        self._collection_name = collection_name
        self._use_chroma = False
        self._store: list[dict[str, Any]] = []
        self._collection = None
        self._next_index = 0

        try:
            import chromadb  # noqa: F401

            # TODO: initialize chromadb client + collection
            client = chromadb.Client()
            try:
                client.delete_collection(name=collection_name)
            except Exception:
                pass
            self._collection = client.create_collection(
                name=collection_name, 
                metadata={"hnsw:space": "cosine"} # Explicitly use Cosine Similarity
            )
            self._use_chroma = True
        except Exception:
            self._use_chroma = False
            self._collection = None

    def _make_record(self, doc: Document) -> dict[str, Any]:
        # TODO: build a normalized stored record for one document
        embedding = self._embedding_fn(doc.content)
        metadata = doc.metadata.copy()
        metadata["doc_id"] = doc.id
        
        self._next_index += 1
        chunk_id = f"{doc.id}_{self._next_index}"
        
        return {
            "id": chunk_id,
            "content": doc.content,
            "embedding": embedding,
            "metadata": metadata,
        }

    def _search_records(self, query: str, records: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        # TODO: run in-memory similarity search over provided records
        query_embedding = self._embedding_fn(query)
        similarities = []
        for record in records:
            similarity = compute_similarity(query_embedding, record["embedding"])
            similarities.append((record, similarity))
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        ans = []
        for record, score in similarities[:top_k]:
            rec = record.copy()
            rec["score"] = score
            ans.append(rec)
        return ans

    def add_documents(self, docs: list[Document]) -> None:
        """
        Embed each document's content and store it.

        For ChromaDB: use collection.add(ids=[...], documents=[...], embeddings=[...])
        For in-memory: append dicts to self._store
        """
        # TODO: embed each doc and add to store
        if self._use_chroma:
            ids = []
            documents = []
            embeddings = []
            metadatas = []
            for doc in docs:
                record = self._make_record(doc)
                ids.append(record["id"])
                documents.append(record["content"])
                embeddings.append(record["embedding"])
                metadatas.append(record["metadata"])
            if ids:
                self._collection.add(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)
        else:
            for doc in docs:
                self._store.append(self._make_record(doc))

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """
        Find the top_k most similar documents to query.

        For in-memory: compute dot product of query embedding vs all stored embeddings.
        """
        # TODO: embed query, compute similarities, return top_k
        if self._use_chroma:
            results = self._collection.query(
                query_embeddings=[self._embedding_fn(query)],
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )
            ret = []
            for i in range(len(results["ids"][0])):
                ret.append({
                    "id": results["ids"][0][i],
                    "content": results["documents"][0][i] if results["documents"] else "",
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    # Chroma 'cosine' distance is (1 - cosine_similarity)
                    # We return the actual cosine_similarity (0.0 to 1.0)
                    "score": 1.0 - results["distances"][0][i] if "distances" in results and results["distances"] else 0.0,
                })
            return ret
        else:
            return self._search_records(query, self._store, top_k)

    def get_collection_size(self) -> int:
        """Return the total number of stored chunks."""
        # TODO
        if self._use_chroma:
            return self._collection.count()
        else:
            return len(self._store)

    def search_with_filter(self, query: str, top_k: int = 3, metadata_filter: dict = None) -> list[dict]:
        """
        Search with optional metadata pre-filtering.

        First filter stored chunks by metadata_filter, then run similarity search.
        """
        # TODO: filter by metadata, then search among filtered chunks
        if self._use_chroma:
            kwargs = {
                "query_embeddings": [self._embedding_fn(query)],
                "n_results": top_k,
                "include": ["documents", "metadatas", "distances"]
            }
            if metadata_filter is not None:
                kwargs["where"] = metadata_filter

            results = self._collection.query(**kwargs)
            ret = []
            for i in range(len(results["ids"][0])):
                ret.append({
                    "id": results["ids"][0][i],
                    "content": results["documents"][0][i] if results["documents"] else "",
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "score": 1.0 - results["distances"][0][i] if "distances" in results and results["distances"] else 0.0,
                })
            return ret
        else:
            filtered_records = [record for record in self._store if metadata_filter is None or all(record["metadata"].get(k) == v for k, v in metadata_filter.items())]
            return self._search_records(query, filtered_records, top_k)

    def delete_document(self, doc_id: str) -> bool:
        """
        Remove all chunks belonging to a document.

        Returns True if any chunks were removed, False otherwise.
        """
        # TODO: remove all stored chunks where metadata['doc_id'] == doc_id
        if self._use_chroma:
            try:
                # Retrieve the existing count
                before_count = self._collection.count()
                self._collection.delete(where={"doc_id": doc_id})
                after_count = self._collection.count()
                return before_count > after_count
            except Exception:
                return False
        else:
            initial_size = len(self._store)
            self._store = [record for record in self._store if record["metadata"].get("doc_id") != doc_id]
            return len(self._store) < initial_size
