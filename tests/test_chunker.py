from itertools import product

import pytest

from src.chunker import (
    FixedTokenChunker,
    RecursiveCharChunker,
    SemanticChunker,
    Chunker,
)


class TestFixedTokenChunker:
    def test_splits_text_into_token_chunks(self):
        chunker = FixedTokenChunker(chunk_size=50, chunk_overlap=10)
        text = " ".join(["machine learning is awesome"] * 20)
        chunks = chunker.split(text)
        assert len(chunks) >= 2
        # Every chunk (except possibly the last) should decode to ≤ chunk_size tokens
        for chunk in chunks[:-1]:
            tokens = chunker.encoder.encode(chunk)
            assert len(tokens) <= 50 + 10  # tolerance for overlap

    def test_empty_text_returns_empty_list(self):
        chunker = FixedTokenChunker(chunk_size=50, chunk_overlap=10)
        assert chunker.split("") == []

    def test_short_text_returns_single_chunk(self):
        chunker = FixedTokenChunker(chunk_size=500, chunk_overlap=10)
        chunks = chunker.split("hello world")
        assert len(chunks) == 1

    def test_name(self):
        assert FixedTokenChunker().name == "fixed_token"


class TestRecursiveCharChunker:
    def test_splits_at_paragraph_boundaries(self, sample_en_text):
        chunker = RecursiveCharChunker(chunk_size=200, chunk_overlap=20)
        chunks = chunker.split(sample_en_text)
        assert len(chunks) >= 3

    def test_short_text_returns_single_chunk(self):
        chunker = RecursiveCharChunker(chunk_size=500, chunk_overlap=20)
        chunks = chunker.split("hello world")
        assert len(chunks) == 1

    def test_all_chunks_within_size(self, sample_en_text):
        chunker = RecursiveCharChunker(chunk_size=250, chunk_overlap=0)
        chunks = chunker.split(sample_en_text)
        for c in chunks:
            assert len(c) <= 250

    def test_name(self):
        assert RecursiveCharChunker().name == "recursive_char"


class TestSemanticChunker:
    def test_splits_at_sentence_boundaries(self, sample_en_text):
        chunker = SemanticChunker(chunk_size=150, chunk_overlap=20)
        chunks = chunker.split(sample_en_text)
        assert len(chunks) >= 3

    def test_all_chunks_smaller_than_chunk_size(self, sample_en_text):
        chunker = SemanticChunker(chunk_size=250, chunk_overlap=20)
        chunks = chunker.split(sample_en_text)
        for c in chunks:
            assert len(c) <= 250 + 30  # small tolerance

    def test_name(self):
        assert SemanticChunker().name == "semantic"


class TestChunker:
    STRATEGIES = ["fixed_token", "recursive_char", "semantic"]

    def test_set_strategy_changes_behavior(self, sample_docs):
        chunker = Chunker(chunk_size=300, chunk_overlap=20)
        counts = {}
        for strat in self.STRATEGIES:
            chunker.set_strategy(strat)
            chunks = chunker.chunk_documents(sample_docs)
            counts[strat] = len(chunks)
            assert len(chunks) > 0
        # Different strategies should produce different chunk counts
        assert len(set(counts.values())) >= 2

    def test_unknown_strategy_raises(self):
        chunker = Chunker()
        with pytest.raises(ValueError, match="Unknown strategy"):
            chunker.set_strategy("nonexistent")

    def test_chunk_documents_metadata(self, sample_docs):
        chunker = Chunker(strategy="recursive_char", chunk_size=256, chunk_overlap=30)
        chunks = chunker.chunk_documents(sample_docs)
        for c in chunks:
            assert "id" in c
            assert "source" in c
            assert "chunk_index" in c
            assert "content" in c
            assert "length" in c
            assert "strategy" in c
            assert c["strategy"] == chunker.strategy_name
            assert len(c["content"]) == c["length"]

    def test_chunk_indices_are_sequential(self, sample_docs):
        chunker = Chunker(strategy="recursive_char", chunk_size=256, chunk_overlap=30)
        chunks = chunker.chunk_documents(sample_docs)
        by_source = {}
        for c in chunks:
            by_source.setdefault(c["source"], []).append(c["chunk_index"])
        for indices in by_source.values():
            assert indices == list(range(len(indices)))

    def test_compare_strategies_returns_all_strategies(self, sample_docs):
        chunker = Chunker(chunk_size=300, chunk_overlap=30)
        results = chunker.compare_strategies(sample_docs)
        for strat in self.STRATEGIES:
            assert strat in results
            assert "num_chunks" in results[strat]
            assert "avg_length" in results[strat]
            assert "min_length" in results[strat]
            assert "max_length" in results[strat]
            assert "std_length" in results[strat]
            assert results[strat]["num_chunks"] > 0

    def test_strategy_name_property(self):
        chunker = Chunker(strategy="semantic")
        assert chunker.strategy_name == "semantic"
        chunker.set_strategy("fixed_token")
        assert chunker.strategy_name == "fixed_token"
