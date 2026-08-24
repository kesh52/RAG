from abc import ABC, abstractmethod

class BaseRetriever(ABC):
    """Abstract base class for document retrievers in the RAG pipeline."""
    
    @abstractmethod
    def vector_search(self, query_vector: list[float], limit: int = 2) -> list[str]:
        """Performs a dense vector similarity search."""
        pass

    @abstractmethod
    def hybrid_search_rrf(self, query: str, query_vector: list[float], limit: int = 10, rrf_k: int = 60) -> list[str]:
        """Performs a dense vector + sparse keyword search combined with Reciprocal Rank Fusion (RRF)."""
        pass

