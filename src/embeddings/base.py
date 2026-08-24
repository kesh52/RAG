from abc import ABC, abstractmethod

class BaseEmbeddingService(ABC):
    """Abstract base class for dense vector embedding generation services."""
    
    @abstractmethod
    def get_dense_embedding(self, text: str) -> list[float]:
        """Generates a dense vector embedding for the input text."""
        pass

