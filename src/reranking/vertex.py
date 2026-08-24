import logging
from google.cloud import discoveryengine_v1 as discoveryengine
from src.reranking.base import BaseReranker

logger = logging.getLogger(__name__)

class VertexReranker(BaseReranker):
    """Service to rerank candidate chunks using the Vertex AI Semantic Ranker."""
    
    def __init__(self, rank_client, project: str, location: str = "global", model_name: str = "semantic-ranker-512@latest"):
        self.rank_client = rank_client
        self.project = project
        self.location = location
        self.model_name = model_name

    def rank_candidates(self, query: str, candidates: list[str], top_n: int = 2) -> list[str]:
        """Stage 2: Cross-encoder reranking using Vertex AI Semantic Ranker."""
        if not candidates:
            logger.debug("No candidates to rerank")
            return []
        
        logger.debug(f"Reranking {len(candidates)} candidates using model '{self.model_name}' (top_n={top_n})")
        records = [
            discoveryengine.RankingRecord(id=str(idx), content=doc_text)
            for idx, doc_text in enumerate(candidates)
        ]
        ranking_config = self.rank_client.ranking_config_path(
            project=self.project,
            location=self.location,
            ranking_config="default_ranking_config"
        )
        request = discoveryengine.RankRequest(
            ranking_config=ranking_config,
            model=self.model_name,
            query=query,
            records=records,
            top_n=top_n
        )
        response = self.rank_client.rank(request=request)
        results = [record.content for record in response.records]
        logger.debug(f"Semantic Ranker completed: returned {len(results)} reranked contexts")
        return results

