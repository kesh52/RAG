import logging
from google.genai import types as genai_types
from src.embeddings.base import BaseEmbeddingService

logger = logging.getLogger(__name__)

class VertexEmbeddingService(BaseEmbeddingService):
    """Service to handle dense vector embedding generation using Vertex AI."""
    
    def __init__(self, client, model_name: str = "text-embedding-005"):
        self.client = client
        self.model_name = model_name

    def get_dense_embedding(self, text: str) -> list[float]:
        """Generates a dense vector embedding using the configured Vertex AI model."""
        logger.debug(f"Generating dense embedding using model '{self.model_name}' for text length: {len(text)}")
        emb_res = self.client.models.embed_content(
            model=self.model_name,
            contents=text,
            config=genai_types.EmbedContentConfig(
                task_type="RETRIEVAL_QUERY",
                output_dimensionality=768,
            ),
        )
        return emb_res.embeddings[0].values

    def get_dense_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Generates dense vector embeddings for a list of texts using Vertex AI."""
        if not texts:
            return []
        logger.debug(f"Generating dense embeddings for {len(texts)} texts using model '{self.model_name}'")
        try:
            emb_res = self.client.models.embed_content(
                model=self.model_name,
                contents=texts,
                config=genai_types.EmbedContentConfig(
                    task_type="RETRIEVAL_QUERY",
                    output_dimensionality=768,
                ),
            )
            return [e.values for e in emb_res.embeddings]
        except Exception as e:
            logger.warning(f"Batch embedding failed ({e}), falling back to item-by-item: {e}")
            return [self.get_dense_embedding(t) for t in texts]

