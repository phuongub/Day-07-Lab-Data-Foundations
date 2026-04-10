from typing import Callable

from .store import EmbeddingStore


class KnowledgeBaseAgent:
    """
    An agent that answers questions using a vector knowledge base.

    Retrieval-augmented generation (RAG) pattern:
        1. Retrieve top-k relevant chunks from the store.
        2. Build a prompt with the chunks as context.
        3. Call the LLM to generate an answer.
    """

    def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str], str]) -> None:
        # TODO: store references to store and llm_fn
        self.store = store
        self.llm_fn = llm_fn

    def answer(self, question: str, top_k: int = 3) -> str:
        # 1. Retrieve top-k relevant chunks from the store
        chunks = self.store.search(question, top_k)
        
        # 2. Build a prompt with the chunks as context
        if not chunks:
            context = "No relevant information found."
        else:
            # Join chunks with a clear separator but without artificial truncation
            context = "\n\n---\n".join([f"[Source: {c['metadata'].get('source', 'Unknown')}]\n{c['content']}" for c in chunks])
        
        prompt = (
            "You are a helpful assistant. Use the context below to answer the question.\n"
            "If the answer is not in the context, say you don't know.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n"
            "Answer:"
        )

        # 3. Call the LLM to generate an answer
        return self.llm_fn(prompt)
