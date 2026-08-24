from abc import ABC, abstractmethod

class BaseReranker(ABC):
    """Abstract base class for Stage-2 document rerankers in the RAG pipeline."""
    
    @abstractmethod
    def rank_candidates(self, query: str, candidates: list[str], top_n: int = 2) -> list[str]:
        """Reranks candidate documents with respect to the query."""
        pass

