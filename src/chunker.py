import re
import logging
import tiktoken
from abc import ABC, abstractmethod
from typing import List, Dict, Optional
from config import config


class ChunkingStrategy(ABC):
    """Base class for text chunking strategies."""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @abstractmethod
    def split(self, text: str) -> List[str]:
        """Split text into chunks."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy name."""


class FixedTokenChunker(ChunkingStrategy):
    """Split text by fixed token count using tiktoken."""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50,
                 model: str = "gpt-3.5-turbo"):
        super().__init__(chunk_size, chunk_overlap)
        try:
            self.encoder = tiktoken.encoding_for_model(model)
        except Exception:
            self.encoder = tiktoken.get_encoding("cl100k_base")

    @property
    def name(self) -> str:
        return "fixed_token"

    def split(self, text: str) -> List[str]:
        tokens = self.encoder.encode(text)
        chunks = []
        start = 0
        while start < len(tokens):
            end = min(start + self.chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            chunks.append(self.encoder.decode(chunk_tokens))
            if end >= len(tokens):
                break
            start = end - self.chunk_overlap
        return chunks


class RecursiveCharChunker(ChunkingStrategy):
    """Recursively split by separators (paragraph, sentence, word)."""

    separators = ["\n\n", "\n", ". ", " ", ""]

    @property
    def name(self) -> str:
        return "recursive_char"

    def split(self, text: str) -> List[str]:
        return self._split_recursive(text, self.separators)

    def _split_recursive(self, text: str, seps: List[str]) -> List[str]:
        sep = seps[0]
        remaining_seps = seps[1:]

        if not sep:
            # Character-level splitting
            chunks = []
            for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
                chunk = text[i:i + self.chunk_size]
                if chunk:
                    chunks.append(chunk)
            return chunks

        splits = text.split(sep)
        chunks = []
        current = ""
        for part in splits:
            test = current + (sep if current else "") + part
            if len(test) <= self.chunk_size:
                current = test
            else:
                if current:
                    chunks.append(current)
                if len(part) > self.chunk_size and remaining_seps:
                    sub_chunks = self._split_recursive(part, remaining_seps)
                    chunks.extend(sub_chunks)
                    current = ""
                else:
                    current = part
        if current:
            chunks.append(current)
        return chunks


class SemanticChunker(ChunkingStrategy):
    """Split by sentences, grouping until reaching chunk size."""

    @property
    def name(self) -> str:
        return "semantic"

    def split(self, text: str) -> List[str]:
        sentences = self._sent_tokenize(text)
        chunks = []
        current = ""
        for sent in sentences:
            if len(current) + len(sent) <= self.chunk_size:
                current += (" " if current else "") + sent
            else:
                if current:
                    chunks.append(current)
                if len(sent) > self.chunk_size:
                    # Long sentence: split at chunk_size
                    for i in range(0, len(sent), self.chunk_size - self.chunk_overlap):
                        chunks.append(sent[i:i + self.chunk_size])
                    current = ""
                else:
                    current = sent
        if current:
            chunks.append(current)
        return chunks

    @staticmethod
    def _sent_tokenize(text: str) -> List[str]:
        return re.split(r'(?<=[.!?])\s+', text)


class Chunker:
    """Main chunker with multiple strategies and metadata tracking."""

    def __init__(self, strategy: str = "recursive_char",
                 chunk_size: int = 512, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._build_strategies()
        self.set_strategy(strategy)

    def _build_strategies(self):
        self.strategies: Dict[str, ChunkingStrategy] = {
            "fixed_token": FixedTokenChunker(self.chunk_size, self.chunk_overlap),
            "recursive_char": RecursiveCharChunker(self.chunk_size, self.chunk_overlap),
            "semantic": SemanticChunker(self.chunk_size, self.chunk_overlap),
        }

    def set_strategy(self, name: str):
        if name not in self.strategies:
            raise ValueError(f"Unknown strategy: {name}. Options: {list(self.strategies.keys())}")
        self._strategy = self.strategies[name]

    @property
    def strategy_name(self) -> str:
        return self._strategy.name

    def chunk_documents(self, docs: List[Dict],
                        strategy: Optional[str] = None) -> List[Dict]:
        """Chunk a list of documents, returning chunk metadata."""
        if strategy:
            self.set_strategy(strategy)

        chunks = []
        for doc in docs:
            texts = self._strategy.split(doc["content"])
            for i, text in enumerate(texts):
                if not text.strip():
                    continue
                chunks.append({
                    "id": f"{doc['source']}_chunk_{i:04d}",
                    "source": doc["source"],
                    "chunk_index": i,
                    "content": text,
                    "length": len(text),
                    "strategy": self.strategy_name,
                })

        logging.getLogger("rag_lab").info(
            "Chunked %d documents into %d chunks "
            "(strategy=%s, size=%d, overlap=%d)",
            len(docs), len(chunks),
            self.strategy_name, self.chunk_size, self.chunk_overlap)
        return chunks

    def compare_strategies(self, docs: List[Dict]) -> Dict[str, dict]:
        """Run all strategies and return statistics for comparison."""
        results = {}
        for name in self.strategies:
            chunks = self.chunk_documents(docs, strategy=name)
            lengths = [c["length"] for c in chunks]
            results[name] = {
                "num_chunks": len(chunks),
                "avg_length": sum(lengths) / len(lengths) if lengths else 0,
                "min_length": min(lengths) if lengths else 0,
                "max_length": max(lengths) if lengths else 0,
                "std_length": (sum((l - sum(lengths)/len(lengths))**2
                                   for l in lengths) / len(lengths))**0.5 if lengths else 0,
            }
        return results
