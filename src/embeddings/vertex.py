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

