import logging
from google import genai
from google.genai import types as genai_types
from google.cloud import discoveryengine_v1 as discoveryengine
import src.db as db
from src.utils.config import config

from src.embeddings.vertex import VertexEmbeddingService
from src.retrieval.postgres import PostgresRetriever
from src.reranking.vertex import VertexReranker
from src.pipeline.prompts import (
    DEFAULT_STANDARD_PROMPT,
    DEFAULT_ATTACHED_REPORT_PROMPT,
    format_prompt,
)

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

    def prepare_retrieval_and_prompt(
        self,
        query: str,
        use_hybrid: bool = True,
        use_reranker: bool = True,
        pool_size: int = 5,
        final_top_k: int = 2,
        attached_file_bytes: bytes | None = None,
        attached_filename: str | None = None,
        attached_mime_type: str | None = None,
        prompt_template: str | None = None,
    ) -> tuple[list[dict], list[str], any]:
        """Performs retrieval, reranking, and formats the multimodal LLM generation prompt."""
        logger.info(f"Initiating RAG retrieval & prompt preparation for query: '{query}' (attached file: {attached_filename})")

        # 1. Prepare search query (augment with attached text if plain text / log file)
        search_query = query
        mime_type = attached_mime_type

        if attached_file_bytes and attached_filename:
            fn_lower = attached_filename.lower()
            if not mime_type:
                if fn_lower.endswith(".pdf"):
                    mime_type = "application/pdf"
                elif fn_lower.endswith((".png", ".bmp")):
                    mime_type = "image/png"
                elif fn_lower.endswith((".jpg", ".jpeg")):
                    mime_type = "image/jpeg"
                elif fn_lower.endswith((".txt", ".log", ".json", ".md")):
                    mime_type = "text/plain"
                else:
                    mime_type = "application/octet-stream"

            # If plain text or log, extract snippet to enrich the search query
            if mime_type.startswith("text/"):
                try:
                    text_snippet = attached_file_bytes.decode("utf-8", errors="ignore")[:600].strip()
                    if text_snippet:
                        search_query = f"{query}\n[Report Snippet: {text_snippet}]"
                except Exception:
                    pass

        # 2. Embed query for dense searches
        query_vector = self.embedding_service.get_dense_embedding(search_query)

        # 3. Stage 1: Retrieval
        if use_hybrid:
            candidates = self.retriever.hybrid_search_rrf(search_query, query_vector, limit=pool_size)
        else:
            candidates = self.retriever.vector_search(query_vector, limit=pool_size)

        # 4. Stage 2: Reranking
        if use_reranker:
            retrieved_contexts = self.reranker.rank_candidates(search_query, candidates, top_n=final_top_k)
        else:
            logger.debug("Semantic reranking is disabled; using top candidates directly")
            retrieved_contexts = candidates[:final_top_k]

        # Extract unique source URLs (prioritizing direct image/PDF file links over parent page URL)
        sources = []
        for doc in retrieved_contexts:
            metadata = doc.get("metadata") or {}
            url = metadata.get("image_url") or metadata.get("pdf_url") or metadata.get("source_url")
            if url and url not in sources:
                sources.append(url)

        # Build context blocks with source prefixing
        context_parts = []
        for doc in retrieved_contexts:
            metadata = doc.get("metadata") or {}
            url = metadata.get("image_url") or metadata.get("pdf_url") or metadata.get("source_url", "Unknown source")
            context_parts.append(f"[Source: {url}]\n{doc['content']}")
        context_block = "\n\n".join(context_parts)

        # Format prompt according to whether an attachment (report/PDF) was provided and template override
        if attached_file_bytes:
            if not mime_type:
                mime_type = "application/pdf"
            template_to_use = prompt_template
            if not template_to_use:
                dynamic_attached_tpl = config.get_dynamic("pipeline.attached_prompt_template")
                template_to_use = dynamic_attached_tpl if dynamic_attached_tpl else DEFAULT_ATTACHED_REPORT_PROMPT
            prompt_instruction = format_prompt(template_to_use, context_block, query)

            llm_contents = [
                genai_types.Part.from_bytes(data=attached_file_bytes, mime_type=mime_type),
                prompt_instruction,
            ]
        else:
            template_to_use = prompt_template
            if not template_to_use:
                dynamic_tpl = config.get_dynamic("pipeline.prompt_template")
                template_to_use = dynamic_tpl if dynamic_tpl else DEFAULT_STANDARD_PROMPT
            llm_contents = format_prompt(template_to_use, context_block, query)

        return retrieved_contexts, sources, llm_contents

    def parse_response_text(self, raw_text: str, sources: list[str]) -> tuple[str, str | None]:
        """Parses generated text to extract documentation gaps tags and append sources footer."""
        import re

        text = (raw_text or "").strip()
        documentation_gaps = None

        gaps_pattern = r"<!--\s*DOCUMENTATION_GAPS\s*-->([\s\S]*?)<!--\s*END_DOCUMENTATION_GAPS\s*-->"
        gaps_match = re.search(gaps_pattern, text, re.IGNORECASE)
        if gaps_match:
            gaps_content = gaps_match.group(1).strip()
            text = re.sub(gaps_pattern, "", text, flags=re.IGNORECASE).strip()
            if gaps_content and not gaps_content.lower().startswith("none"):
                documentation_gaps = gaps_content

        # Also strip any legacy inline gap headers from user response if generated
        legacy_gap_pattern = r"(?:###\s*(?:\d+\.\s*)?⚠️\s*Internal Documentation Gaps[\s\S]*?)(?=(?:###|\Z))"
        legacy_match = re.search(legacy_gap_pattern, text, re.IGNORECASE)
        if legacy_match:
            legacy_block = legacy_match.group(0).strip()
            legacy_body = re.sub(r"^###\s*(?:\d+\.\s*)?⚠️\s*Internal Documentation Gaps\s*", "", legacy_block, flags=re.IGNORECASE).strip()
            text = text.replace(legacy_block, "").strip()
            if legacy_body and not legacy_body.lower().startswith("none") and not documentation_gaps:
                documentation_gaps = legacy_body

        # Append reference links footer to the final response text if sources exist
        if sources and "Sources:" not in text:
            sources_footer = "\n\nSources:\n" + "\n".join(f"- {src}" for src in sources)
            text += sources_footer

        return text, documentation_gaps

    def retrieve_and_generate_stream(
        self,
        query: str,
        use_hybrid: bool = True,
        use_reranker: bool = True,
        pool_size: int = 5,
        final_top_k: int = 2,
        attached_file_bytes: bytes | None = None,
        attached_filename: str | None = None,
        attached_mime_type: str | None = None,
        model_name: str | None = None,
        prompt_template: str | None = None,
    ):
        """Retrieves contexts and returns a streaming generator from Gemini along with metadata."""
        retrieved_contexts, sources, llm_contents = self.prepare_retrieval_and_prompt(
            query=query,
            use_hybrid=use_hybrid,
            use_reranker=use_reranker,
            pool_size=pool_size,
            final_top_k=final_top_k,
            attached_file_bytes=attached_file_bytes,
            attached_filename=attached_filename,
            attached_mime_type=attached_mime_type,
            prompt_template=prompt_template,
        )
        gen_model = model_name or self.generator_model
        logger.debug(f"Streaming response using model '{gen_model}' with {len(retrieved_contexts)} context chunks...")
        stream_res = self.generator_client.models.generate_content_stream(
            model=gen_model,
            contents=llm_contents,
        )
        return stream_res, retrieved_contexts, sources

    def retrieve_and_generate(
        self,
        query: str,
        use_hybrid: bool = True,
        use_reranker: bool = True,
        pool_size: int = 5,
        final_top_k: int = 2,
        attached_file_bytes: bytes | None = None,
        attached_filename: str | None = None,
        attached_mime_type: str | None = None,
        model_name: str | None = None,
        prompt_template: str | None = None,
    ) -> dict:
        """End-to-end RAG pipeline supporting dynamic feature toggles and multimodal document attachments."""
        retrieved_contexts, sources, llm_contents = self.prepare_retrieval_and_prompt(
            query=query,
            use_hybrid=use_hybrid,
            use_reranker=use_reranker,
            pool_size=pool_size,
            final_top_k=final_top_k,
            attached_file_bytes=attached_file_bytes,
            attached_filename=attached_filename,
            attached_mime_type=attached_mime_type,
            prompt_template=prompt_template,
        )

        gen_model = model_name or self.generator_model
        logger.debug(f"Generating response using model '{gen_model}' with {len(retrieved_contexts)} context chunks...")
        gen_res = self.generator_client.models.generate_content(
            model=gen_model,
            contents=llm_contents,
        )

        response_text, documentation_gaps = self.parse_response_text(gen_res.text or "", sources)
        logger.info("RAG pipeline execution completed successfully.")
        return {
            "user_input": query,
            "retrieved_contexts": [doc["content"] for doc in retrieved_contexts],
            "sources": sources,
            "response": response_text,
            "attached_filename": attached_filename,
            "documentation_gaps": documentation_gaps,
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
        model_name=config.get_dynamic("models.embedding", config.get("models.embedding", "text-embedding-005"))
    )
    retriever = PostgresRetriever(db.get_connection)
    reranker = VertexReranker(
        rank_client, 
        gcp_project, 
        model_name=config.get_dynamic("models.rerank", config.get("models.rerank", "semantic-ranker-512@latest"))
    )

    return RAGPipeline(
        embedding_service=embedding_service,
        retriever=retriever,
        reranker=reranker,
        generator_client=genai_client,
        generator_model=config.get_dynamic("models.generation", config.get("models.generation", "gemini-2.5-flash"))
    )

