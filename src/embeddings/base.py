from abc import ABC, abstractmethod

class BaseEmbeddingService(ABC):
    """Abstract base class for dense vector embedding generation services."""
    
    @abstractmethod
    def get_dense_embedding(self, text: str) -> list[float]:
        """Generates a dense vector embedding for the input text."""
        pass

    def get_dense_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Generates dense vector embeddings for a batch of input texts."""
        return [self.get_dense_embedding(t) for t in texts]

