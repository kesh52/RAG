import pytest
import numpy as np
from unittest.mock import MagicMock
from src.etl.chunking import (
    BaseChunker,
    RecursiveTextChunker,
    SemanticChunker,
    get_chunker,
)
from src.embeddings.base import BaseEmbeddingService


class MockEmbeddingService(BaseEmbeddingService):
    def __init__(self, mapping=None, default_dim=16):
        self.mapping = mapping or {}
        self.default_dim = default_dim

    def get_dense_embedding(self, text: str) -> list[float]:
        for k, v in self.mapping.items():
            if k.lower() in text.lower():
                return v
        return [0.1] * self.default_dim


def test_recursive_chunker_basic():
    chunker = RecursiveTextChunker(chunk_size=20, chunk_overlap=5)
    text = 'Sentence one here. Sentence two here. Sentence three here.'
    chunks = chunker.split_text(text)
    assert len(chunks) >= 2
    assert all(len(c) <= 20 for c in chunks)


def test_recursive_chunker_details():
    chunker = RecursiveTextChunker(chunk_size=50, chunk_overlap=10)
    details = chunker.split_text_with_details('Hello world. This is a test.')
    assert details['strategy'] == 'recursive'
    assert 'chunks' in details
    assert details['total_chunks'] == len(details['chunks'])


def test_semantic_chunker_empty_and_single():
    chunker = SemanticChunker(min_chunk_size=10, max_chunk_size=500)
    assert chunker.split_text('') == []
    assert chunker.split_text('   ') == []
    single = chunker.split_text('Only one sentence.')
    assert single == ['Only one sentence.']


def test_semantic_chunker_fallback_without_service():
    chunker = SemanticChunker(embedding_service=None, min_chunk_size=10)
    text = 'Sentence one. Sentence two. Sentence three.'
    chunks = chunker.split_text(text)
    assert len(chunks) >= 1
    assert 'Sentence one.' in chunks[0]


def test_semantic_chunker_cosine_breakpoint_detection():
    # Construct 2 distinct topics with orthogonal embeddings:
    # Topic A: [1, 0, 0, 0]
    # Topic B: [0, 1, 0, 0]
    mapping = {
        'cats': [1.0, 0.0, 0.0, 0.0],
        'feline': [1.0, 0.0, 0.0, 0.0],
        'quantum': [0.0, 1.0, 0.0, 0.0],
        'physics': [0.0, 1.0, 0.0, 0.0],
    }
    emb = MockEmbeddingService(mapping=mapping, default_dim=4)

    chunker = SemanticChunker(
        embedding_service=emb,
        breakpoint_threshold_type='fixed',
        breakpoint_threshold_amount=0.5,
        buffer_size=0,
        min_chunk_size=5,
        max_chunk_size=500,
    )

    text = 'All cats are cute. Felines purr loudly. Quantum entanglement is fascinating. Theoretical physics explains particles.'
    details = chunker.split_text_with_details(text)

    chunks = details['chunks']
    assert len(chunks) == 2
    assert 'cats' in chunks[0] and 'purr' in chunks[0]
    assert 'Quantum' in chunks[1] and 'physics' in chunks[1]
    assert details['threshold'] == 0.5
    assert len(details['breakpoint_indices']) == 1


def test_semantic_chunker_threshold_types():
    mapping = {
        'alpha': [1.0, 0.0, 0.0, 0.0],
        'beta': [0.9, 0.1, 0.0, 0.0],
        'gamma': [0.0, 1.0, 0.0, 0.0],
        'delta': [0.0, 0.9, 0.1, 0.0],
    }
    emb = MockEmbeddingService(mapping=mapping, default_dim=4)
    text = 'Topic alpha begins. Topic beta continues. Topic gamma shifts completely. Topic delta concludes.'

    for ttype in ['percentile', 'standard_deviation', 'interquartile', 'gradient', 'fixed']:
        amount = 0.4 if ttype == 'fixed' else 75.0 if ttype in ['percentile', 'gradient'] else 0.5
        chunker = SemanticChunker(
            embedding_service=emb,
            breakpoint_threshold_type=ttype,
            breakpoint_threshold_amount=amount,
            buffer_size=0,
            min_chunk_size=5,
            max_chunk_size=1000,
        )
        details = chunker.split_text_with_details(text)
        assert len(details['chunks']) >= 1
        assert details['threshold_type'] == ttype


def test_semantic_chunker_size_constraints():
    # Verify min_chunk_size merging
    emb = MockEmbeddingService()
    chunker = SemanticChunker(
        embedding_service=emb,
        min_chunk_size=100,
        max_chunk_size=500,
    )
    text = 'Short A. Short B. Short C.'
    chunks = chunker.split_text(text)
    # Because each is under 100 chars, they should be merged
    assert len(chunks) == 1


def test_get_chunker_factory():
    c_rec = get_chunker('recursive', chunk_size=300, chunk_overlap=30)
    assert isinstance(c_rec, RecursiveTextChunker)
    assert c_rec.chunk_size == 300
    assert c_rec.chunk_overlap == 30

    emb = MockEmbeddingService()
    c_sem = get_chunker('semantic', embedding_service=emb, breakpoint_threshold_amount=95.0)
    assert isinstance(c_sem, SemanticChunker)
    assert c_sem.breakpoint_threshold_amount == 95.0

    with pytest.raises(ValueError, match='Unknown chunking strategy'):
        get_chunker('unknown_strategy')
