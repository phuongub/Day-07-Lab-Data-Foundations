from __future__ import annotations

import os
import sys
from pathlib import Path

QUERY = "VinFast có hỗ trợ đổi xe máy điện cũ sang ô tô điện VF 3 không?"
os.environ["HF_HOME"] = str(Path("./.hf-cache").resolve())

if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

from dotenv import load_dotenv

from src.agent import KnowledgeBaseAgent
from src.embeddings import (
    EMBEDDING_PROVIDER_ENV,
    LOCAL_EMBEDDING_MODEL,
    OPENAI_EMBEDDING_MODEL,
    LocalEmbedder,
    OpenAIEmbedder,
    _mock_embed,
)
from src.models import Document
from src.store import EmbeddingStore
from src.chunking import RecursiveChunker, ChunkingStrategyComparator, SentenceChunker, FixedSizeChunker

SAMPLE_DIR = Path("data")
SAMPLE_FILES = [str(p) for p in SAMPLE_DIR.glob("*.md")] if SAMPLE_DIR.exists() else []


def load_documents_from_files(file_paths: list[str]) -> list[Document]:
    """Load documents from file paths for the manual demo."""
    allowed_extensions = {".md", ".txt"}
    documents: list[Document] = []

    for raw_path in file_paths:
        path = Path(raw_path)

        if path.suffix.lower() not in allowed_extensions:
            print(f"Skipping unsupported file type: {path} (allowed: .md, .txt)")
            continue

        if not path.exists() or not path.is_file():
            print(f"Skipping missing file: {path}")
            continue

        content = path.read_text(encoding="utf-8")
        documents.append(
            Document(
                id=path.stem,
                content=content,
                metadata={"source": str(path), "extension": path.suffix.lower()},
            )
        )

    return documents


def generate_chunking_report(docs: list[Document], query: str, embedder: any, output_path: str = "results_Q4.md"):
    """Compares retrieval performance of different strategies and saves to a file."""
    report = ["# Chunking Strategy Comparison Report\n"]
    report.append(f"**Test Question:** {query}\n")
    report.append(f"Total Documents: {len(docs)}\n")
    report.append("| Strategy | Total Chunks | Avg. Length | Top Match Score |")
    report.append("| :--- | :--- | :--- | :--- |")
    
    strategies = [
        ("Fixed-Size", FixedSizeChunker(chunk_size=500)),
        ("Sentence-Based", SentenceChunker()),
        ("Recursive", RecursiveChunker(chunk_size=1000))
    ]

    method_details = []

    for label, chunker in strategies:
        # Create a temporary store for this strategy
        temp_store = EmbeddingStore(collection_name=f"temp_{label.lower()}", embedding_fn=embedder)
        all_chunks = []
        total_len = 0
        
        for doc in docs:
            segments = chunker.chunk(doc.content)
            for i, seg in enumerate(segments):
                total_len += len(seg)
                all_chunks.append(Document(id=f"{doc.id}_{i}", content=seg, metadata=doc.metadata))
        
        temp_store.add_documents(all_chunks)
        results = temp_store.search(query, top_k=3)
        
        avg_len = total_len / len(all_chunks) if all_chunks else 0
        top_score = results[0]["score"] if results else 0
        report.append(f"| {label} | {len(all_chunks)} | {avg_len:.1f} | {format_score(top_score)} |")
        
        # Save details for the section below
        method_details.append((label, results))

    report.append("\n## Top 3 Retrieved Chunks per Strategy\n")
    for label, results in method_details:
        report.append(f"### Strategy: {label}")
        if not results:
            report.append("_No results found._\n")
            continue
            
        for i, res in enumerate(results, 1):
            source = res["metadata"].get("source", "unknown")
            report.append(f"**{i}. [Score: {format_score(res['score'])}] Source: {source}**")
            content = res["content"].replace("\n", " ")[:300]
            report.append(f"> {content}...\n")
        report.append("---\n")

    Path(output_path).write_text("\n".join(report), encoding="utf-8")
    print(f"\n[REPORT GENERATED] Strategy comparison for '{query}' saved to {output_path}")


def format_score(score: float) -> str:
    """Converts a cosine similarity score (0.0 to 1.0) into a display string."""
    # Since we switched Chroma to 'cosine' space, the score is already the similarity
    confidence = max(0.0, min(1.0, score)) * 100
    return f"{score:.3f} ({confidence:.1f}%)"


def demo_llm(prompt: str) -> str:
    """A cleaner mock LLM output that focuses on the found info."""
    separator = "Context:\n"
    content_after_header = prompt.split(separator)[-1] if separator in prompt else prompt
    
    # Take a larger slice but skip the 'You are a helpful assistant' part
    clean_output = content_after_header[:1500].strip()
    return f"\n--- [AGENT RESPONSE SIMULATION] ---\n{clean_output}\n\n[... Output continues ...]\n---"


def run_manual_demo(question: str | None = None, sample_files: list[str] | None = None) -> int:
    files = sample_files or SAMPLE_FILES
    # Use CLI question if available, otherwise fallback to the QUERY variable at the top
    query = question or QUERY

    print("=== Manual File Test ===")
    print("Accepted file types: .md, .txt")
    print("Input file list:")
    for file_path in files:
        print(f"  - {file_path}")

    docs = load_documents_from_files(files)
    if not docs:
        print("\nNo valid input files were loaded.")
        print("Create files matching the sample paths above, then rerun:")
        print("  python3 main.py")
        return 1

    print(f"\nLoaded {len(docs)} documents")
    for doc in docs:
        print(f"  - {doc.id}: {doc.metadata['source']}")

    print("\nInitializing embedding backend...")
    try:
        # Check if you have a local 'models' folder as you mentioned
        local_model_path = Path("models/all-MiniLM-L6-v2")
        if local_model_path.exists():
            print(f"Loading model from local directory: {local_model_path}")
            embedder = LocalEmbedder(model_name=str(local_model_path.resolve()))
        else:
            # Fallback to standard name (uses .hf-cache)
            embedder = LocalEmbedder()
    except Exception as e:
        print(f"Local embedder unavailable ({e}), using mock fallback.")
        embedder = _mock_embed

    print(f"\nEmbedding backend: {getattr(embedder, '_backend_name', embedder.__class__.__name__)}")
    
    # NEW: Generate the comparison report for all documents using the REAL embedder
    generate_chunking_report(docs, query, embedder)

    store = EmbeddingStore(collection_name="manual_test_store", embedding_fn=embedder)
    
    # Chunk documents before adding to store
    chunker = SentenceChunker()
    all_chunks = []
    for doc in docs:
        segments = chunker.chunk(doc.content)
        for i, segment in enumerate(segments):
            all_chunks.append(
                Document(
                    id=f"{doc.id}_chunk_{i}",
                    content=segment,
                    metadata={**doc.metadata, "chunk_index": i}
                )
            )
    
    store.add_documents(all_chunks)

    print(f"\nStored {store.get_collection_size()} chunks in EmbeddingStore (from {len(docs)} documents)")
    print("\n=== EmbeddingStore Search Test ===")
    print(f"Query: {query}")
    search_results = store.search(query, top_k=3)
    for index, result in enumerate(search_results, start=1):
        print(f"{index}. score={format_score(result['score'])} source={result['metadata'].get('source')}")
        print(f"   content preview: {result['content'][:120].replace(chr(10), ' ')}...")

    print("\n=== KnowledgeBaseAgent Test ===")
    agent = KnowledgeBaseAgent(store=store, llm_fn=demo_llm)
    print(f"Question: {query}")
    print("Agent answer:")
    print(agent.answer(query, top_k=2))
    return 0


def main() -> int:
    question = " ".join(sys.argv[1:]).strip() if len(sys.argv) > 1 else None
    return run_manual_demo(question=question)


if __name__ == "__main__":
    raise SystemExit(main())
