import logging
from google import genai
from google.genai import types as genai_types
from google.cloud import discoveryengine_v1 as discoveryengine
import src.db as db
from src.utils.config import config

from src.embeddings.vertex import VertexEmbeddingService
from src.retrieval.postgres import PostgresRetriever
from src.reranking.vertex import VertexReranker

logger = logging.getLogger(__name__)


class RAGPipeline:
    """Orchestrator class for the modular RAG pipeline."""
    
    def __init__(
        self,
        embedding_service,
        retriever,
        reranker,
        generator_client: genai.Client,
        generator_model: str = "gemini-2.5-flash"
    ):
        self.embedding_service = embedding_service
        self.retriever = retriever
        self.reranker = reranker
        self.generator_client = generator_client
        self.generator_model = generator_model

    def retrieve_and_generate(
        self,
        query: str,
        use_hybrid: bool = True,
        use_reranker: bool = True,
        pool_size: int = 5,
        final_top_k: int = 2
    ) -> dict:
        """End-to-end RAG pipeline supporting dynamic feature toggles."""
        logger.info(f"Initiating RAG pipeline execution for query: '{query}'")
        
        # 1. Embed query for dense searches
        query_vector = self.embedding_service.get_dense_embedding(query)

        # 2. Stage 1: Retrieval
        if use_hybrid:
            candidates = self.retriever.hybrid_search_rrf(query, query_vector, limit=pool_size)
        else:
            candidates = self.retriever.vector_search(query_vector, limit=pool_size)

        # 3. Stage 2: Reranking
        if use_reranker:
            retrieved_contexts = self.reranker.rank_candidates(query, candidates, top_n=final_top_k)
        else:
            logger.debug("Semantic reranking is disabled; using top candidates directly")
            retrieved_contexts = candidates[:final_top_k]

        # 4. Stage 3: Answer Generation
        logger.debug(f"Generating response using model '{self.generator_model}' with {len(retrieved_contexts)} context chunks...")
        
        # Build context blocks with source prefixing
        context_parts = []
        for doc in retrieved_contexts:
            metadata = doc.get("metadata") or {}
            url = metadata.get("image_url") or metadata.get("pdf_url") or metadata.get("source_url", "Unknown source")
            context_parts.append(f"[Source: {url}]\n{doc['content']}")
        context_block = "\n\n".join(context_parts)

        prompt = f"""Context:
        {context_block}

        Question: {query}
        Answer the question concisely and in complete sentences, focusing strictly and directly on the specific item asked. Ignore any unrelated topics or adjacent information present in the context. Inline cite the source URLs for your statements where appropriate:"""

        gen_res = self.generator_client.models.generate_content(
            model=self.generator_model,
            contents=prompt,
        )

        # Extract unique source URLs (prioritizing direct image/PDF file links over parent page URL)
        sources = []
        for doc in retrieved_contexts:
            metadata = doc.get("metadata") or {}
            url = metadata.get("image_url") or metadata.get("pdf_url") or metadata.get("source_url")
            if url and url not in sources:
                sources.append(url)

        # Append reference links footer to the final response text
        response_text = gen_res.text.strip()
        if sources:
            sources_footer = "\n\nSources:\n" + "\n".join(f"- {src}" for src in sources)
            response_text += sources_footer

        logger.info("RAG pipeline execution completed successfully.")
        return {
            "user_input": query,
            "retrieved_contexts": [doc["content"] for doc in retrieved_contexts],
            "sources": sources,
            "response": response_text,
        }


def get_default_pipeline() -> RAGPipeline:
    """Factory helper to instantiate a default RAGPipeline from config configuration."""
    logger.debug("Initializing default RAG pipeline instance from configuration...")
    gcp_project = config.get("gcp.project")
    gcp_location = config.get("gcp.location")

    if not gcp_project or not gcp_location:
        raise ValueError("gcp.project and gcp.location must be configured in config.yaml or environment.")

    genai_client = genai.Client(vertexai=True, project=gcp_project, location=gcp_location)
    rank_client = discoveryengine.RankServiceClient()

    embedding_service = VertexEmbeddingService(
        genai_client, 
        model_name=config.get("models.embedding", "text-embedding-005")
    )
    retriever = PostgresRetriever(db.get_connection)
    reranker = VertexReranker(
        rank_client, 
        gcp_project, 
        model_name=config.get("models.rerank", "semantic-ranker-512@latest")
    )

    return RAGPipeline(
        embedding_service=embedding_service,
        retriever=retriever,
        reranker=reranker,
        generator_client=genai_client,
        generator_model=config.get("models.generation", "gemini-2.5-flash")
    )

