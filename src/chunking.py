from __future__ import annotations

import math
import re


class FixedSizeChunker:
    """
    Split text into fixed-size chunks with optional overlap.

    Rules:
        - Each chunk is at most chunk_size characters long.
        - Consecutive chunks share overlap characters.
        - The last chunk contains whatever remains.
        - If text is shorter than chunk_size, return [text].
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        step = self.chunk_size - self.overlap
        chunks: list[str] = []
        for start in range(0, len(text), step):
            chunk = text[start : start + self.chunk_size]
            chunks.append(chunk)
            if start + self.chunk_size >= len(text):
                break
        return chunks


class SentenceChunker:
    """
    Split text into chunks of at most max_sentences_per_chunk sentences.

    Sentence detection: split on ". ", "! ", "? " or ".\n".
    Strip extra whitespace from each chunk.
    """

    def __init__(self, max_sentences_per_chunk: int = 3) -> None:
        self.max_sentences_per_chunk = max(1, max_sentences_per_chunk)

    def chunk(self, text: str) -> list[str]:
        # Split into sentences using punctuation OR newlines as boundaries
        # We use a non-capturing group to split while keeping the structure relatively clean
        sentences = re.split(r'(?<=[.!?])\s+|\n+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        chunks: list[str] = []
        for i in range(0, len(sentences), self.max_sentences_per_chunk):
            chunk = ' '.join(sentences[i : i + self.max_sentences_per_chunk])
            chunks.append(chunk)
        return chunks


class RecursiveChunker:
    """
    Recursively split text using separators in priority order.

    Default separator priority:
        ["\n\n", "\n", ". ", " ", ""]
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(self, separators: list[str] | None = None, chunk_size: int = 500) -> None:
        self.separators = self.DEFAULT_SEPARATORS if separators is None else list(separators)
        self.chunk_size = chunk_size

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        return self._recursive_split(text, self.separators)

    def _recursive_split(self, text: str, separators: list[str]) -> list[str]:
        # If text is small enough, we are done
        if len(text) <= self.chunk_size:
            return [text]
        
        # If no more separators, we just cut at chunk_size
        if not separators:
            return [text[i:i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]

        # Try to split by current separator
        separator = separators[0]
        splits = [s for s in text.split(separator) if s]
        
        final_chunks = []
        current_chunk = ""
        
        for part in splits:
            # If a single part is still too big, we split it using the next separator level
            sub_parts = self._recursive_split(part, separators[1:])
            
            for sub_part in sub_parts:
                # If adding this sub-part + separator exceeds size, save current and restart
                potential_len = len(current_chunk) + len(sub_part) + (len(separator) if current_chunk else 0)
                
                if potential_len > self.chunk_size and current_chunk:
                    final_chunks.append(current_chunk)
                    current_chunk = sub_part
                else:
                    if current_chunk:
                        current_chunk += separator + sub_part
                    else:
                        current_chunk = sub_part
            
        if current_chunk:
            final_chunks.append(current_chunk)
            
        return final_chunks


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def compute_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    cosine_similarity = dot(a, b) / (||a|| * ||b||)

    Returns 0.0 if either vector has zero magnitude.
    """
    # TODO: implement cosine similarity formula
    dot_product = _dot(vec_a, vec_b)
    norm_a = math.sqrt(_dot(vec_a, vec_a))
    norm_b = math.sqrt(_dot(vec_b, vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)


class ChunkingStrategyComparator:
    """Run all built-in chunking strategies and compare their results."""

    def compare(self, text: str, chunk_size: int = 200) -> dict:
        # TODO: call each chunker, compute stats, return comparison dict
        results = {}
        
        chunks_fixed = FixedSizeChunker(chunk_size=chunk_size).chunk(text)
        results['fixed_size'] = {
            'count': len(chunks_fixed),
            'avg_length': sum(len(c) for c in chunks_fixed) / len(chunks_fixed) if chunks_fixed else 0,
            'chunks': chunks_fixed
        }
        
        chunks_sent = SentenceChunker(max_sentences_per_chunk=3).chunk(text)
        results['by_sentences'] = {
            'count': len(chunks_sent),
            'avg_length': sum(len(c) for c in chunks_sent) / len(chunks_sent) if chunks_sent else 0,
            'chunks': chunks_sent
        }
        
        chunks_rec = RecursiveChunker(chunk_size=chunk_size).chunk(text)
        results['recursive'] = {
            'count': len(chunks_rec),
            'avg_length': sum(len(c) for c in chunks_rec) / len(chunks_rec) if chunks_rec else 0,
            'chunks': chunks_rec
        }
        
        return results
