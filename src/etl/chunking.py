"""Text chunking strategies for the RAG pipeline.

Supports:
1. RecursiveTextChunker: Hierarchical delimiter splitting with token/character overlap.
2. SemanticChunker: Sentence boundary segmentation using embedding distance breakpoint detection.
3. get_chunker: Factory method to instantiate chunkers dynamically.
"""

from abc import ABC, abstractmethod
import logging
import re
from typing import Any, Optional
import numpy as np

from src.embeddings.base import BaseEmbeddingService
from src.utils.config import config

logger = logging.getLogger(__name__)


class BaseChunker(ABC):
    """Abstract base class defining the text chunker interface."""

    @abstractmethod
    def split_text(self, text: str) -> list[str]:
        """Split a long text string into smaller text segments."""
        pass

    def split_text_with_details(self, text: str) -> dict[str, Any]:
        """Split text and return detailed diagnostic/visualization metadata."""
        chunks = self.split_text(text)
        return {
            "chunks": chunks,
            "total_chunks": len(chunks),
            "strategy": self.__class__.__name__,
        }


class RecursiveTextChunker(BaseChunker):
    """Hierarchical text splitter that splits recursively by paragraphs, sentences, and words."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer.")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be non-negative and smaller than chunk_size.")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = ["\n\n", "\n", " ", ""]

    def split_text(self, text: str) -> list[str]:
        cleaned_text = text.strip()
        if not cleaned_text:
            return []
        return self._split(cleaned_text, self.separators)

    def split_text_with_details(self, text: str) -> dict[str, Any]:
        chunks = self.split_text(text)
        return {
            "chunks": chunks,
            "total_chunks": len(chunks),
            "strategy": "recursive",
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
        }

    def _split(self, text: str, separators: list[str]) -> list[str]:
        if len(text) <= self.chunk_size:
            return [text]

        active_separator = ""
        next_separators = []
        for i, sep in enumerate(separators):
            if sep == "" or sep in text:
                active_separator = sep
                next_separators = separators[i + 1 :]
                break

        if active_separator == "":
            splits = list(text)
        else:
            splits = text.split(active_separator)

        final_splits = []
        for part in splits:
            if len(part) > self.chunk_size:
                if next_separators:
                    final_splits.extend(self._split(part, next_separators))
                else:
                    for i in range(0, len(part), self.chunk_size - self.chunk_overlap):
                        final_splits.append(part[i : i + self.chunk_size])
            else:
                final_splits.append(part)

        return self._merge_splits(final_splits, active_separator)

    def _merge_splits(self, splits: list[str], separator: str) -> list[str]:
        docs = []
        current_doc = []
        total_len = 0

        for s in splits:
            d_len = len(s)
            sep_len = len(separator) if current_doc else 0
            if current_doc and total_len + d_len + sep_len > self.chunk_size:
                docs.append(separator.join(current_doc))

                while current_doc:
                    popped = current_doc[0]
                    next_len = total_len - len(popped) - (len(separator) if len(current_doc) > 1 else 0)

                    if next_len + d_len + (len(separator) if len(current_doc) > 1 else 0) <= self.chunk_size and next_len <= self.chunk_overlap:
                        current_doc.pop(0)
                        total_len = next_len
                        break
                    else:
                        current_doc.pop(0)
                        total_len = next_len

            current_doc.append(s)
            total_len += d_len + (len(separator) if len(current_doc) > 1 else 0)

        if current_doc:
            docs.append(separator.join(current_doc))

        return docs


class SemanticChunker(BaseChunker):
    """Semantic boundary text chunker.

    Splits text into sentences, computes contextual embeddings, and finds breakpoints
    where cosine distance between adjacent sentence groups exceeds a statistical threshold.
    """

    SUPPORTED_THRESHOLDS = ["percentile", "standard_deviation", "interquartile", "gradient", "fixed"]

    def __init__(
        self,
        embedding_service: Optional[BaseEmbeddingService] = None,
        breakpoint_threshold_type: str = "percentile",
        breakpoint_threshold_amount: float = 90.0,
        buffer_size: int = 1,
        min_chunk_size: int = 50,
        max_chunk_size: int = 2000,
        sentence_split_regex: Optional[str] = None,
    ):
        if breakpoint_threshold_type not in self.SUPPORTED_THRESHOLDS:
            raise ValueError(
                f"Unsupported breakpoint_threshold_type '{breakpoint_threshold_type}'. "
                f"Supported: {self.SUPPORTED_THRESHOLDS}"
            )
        if min_chunk_size < 0:
            raise ValueError("min_chunk_size must be non-negative.")
        if max_chunk_size <= min_chunk_size:
            raise ValueError("max_chunk_size must be strictly greater than min_chunk_size.")

        self.embedding_service = embedding_service
        self.breakpoint_threshold_type = breakpoint_threshold_type
        self.breakpoint_threshold_amount = breakpoint_threshold_amount
        self.buffer_size = max(0, buffer_size)
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.sentence_split_regex = sentence_split_regex or r"(?<=[.?!])\s+(?=[A-Z0-9\"'`\n])|(?<=\n)\n+"

    def split_text(self, text: str) -> list[str]:
        """Split text semantically into coherent chunks."""
        details = self.split_text_with_details(text)
        return details["chunks"]

    def split_text_with_details(self, text: str) -> dict[str, Any]:
        """Split text and return rich diagnostic metadata for UI visualization."""
        cleaned_text = text.strip()
        if not cleaned_text:
            return {
                "chunks": [],
                "sentences": [],
                "distances": [],
                "threshold": 0.0,
                "breakpoint_indices": [],
                "total_chunks": 0,
                "strategy": "semantic",
                "threshold_type": self.breakpoint_threshold_type,
                "threshold_amount": self.breakpoint_threshold_amount,
            }

        # 1. Split text into individual sentences/propositions
        sentences = self._split_into_sentences(cleaned_text)
        if len(sentences) <= 1:
            return {
                "chunks": [cleaned_text],
                "sentences": sentences,
                "distances": [],
                "threshold": 0.0,
                "breakpoint_indices": [],
                "total_chunks": 1,
                "strategy": "semantic",
                "threshold_type": self.breakpoint_threshold_type,
                "threshold_amount": self.breakpoint_threshold_amount,
            }

        # If no embedding service is provided, fallback to sentence grouping
        if self.embedding_service is None:
            logger.warning("No embedding_service provided to SemanticChunker. Falling back to sentence grouping.")
            fallback_chunks = self._fallback_sentence_chunking(sentences)
            return {
                "chunks": fallback_chunks,
                "sentences": sentences,
                "distances": [],
                "threshold": 0.0,
                "breakpoint_indices": [],
                "total_chunks": len(fallback_chunks),
                "strategy": "semantic_fallback",
                "threshold_type": self.breakpoint_threshold_type,
                "threshold_amount": self.breakpoint_threshold_amount,
            }

        # 2. Build buffered context sentences
        buffered_sentences = self._create_buffered_sentences(sentences)

        # 3. Generate embeddings
        embeddings = self.embedding_service.get_dense_embeddings(buffered_sentences)

        # 4. Compute cosine distances between consecutive sentences
        distances = self._calculate_cosine_distances(embeddings)

        if not distances:
            return {
                "chunks": [cleaned_text],
                "sentences": sentences,
                "distances": [],
                "threshold": 0.0,
                "breakpoint_indices": [],
                "total_chunks": 1,
                "strategy": "semantic",
                "threshold_type": self.breakpoint_threshold_type,
                "threshold_amount": self.breakpoint_threshold_amount,
            }

        # 5. Calculate threshold & identify breakpoints
        threshold = self._calculate_threshold(distances)
        breakpoint_indices = self._identify_breakpoints(distances, threshold)

        # 6. Group sentences into raw chunks
        raw_chunks = self._group_sentences_into_chunks(sentences, breakpoint_indices)

        # 7. Enforce min_chunk_size and max_chunk_size bounds
        final_chunks = self._enforce_size_constraints(raw_chunks)

        return {
            "chunks": final_chunks,
            "sentences": sentences,
            "distances": distances,
            "threshold": threshold,
            "breakpoint_indices": breakpoint_indices,
            "total_chunks": len(final_chunks),
            "strategy": "semantic",
            "threshold_type": self.breakpoint_threshold_type,
            "threshold_amount": self.breakpoint_threshold_amount,
            "buffer_size": self.buffer_size,
        }

    def _split_into_sentences(self, text: str) -> list[str]:
        """Splits raw text into a list of sentence segments."""
        raw_splits = re.split(self.sentence_split_regex, text)
        sentences = [s.strip() for s in raw_splits if s and s.strip()]
        return sentences

    def _create_buffered_sentences(self, sentences: list[str]) -> list[str]:
        """Combines adjacent sentences within buffer_size to smooth local noise."""
        if self.buffer_size == 0:
            return sentences

        buffered = []
        n = len(sentences)
        for i in range(n):
            start_idx = max(0, i - self.buffer_size)
            end_idx = min(n, i + self.buffer_size + 1)
            combined = " ".join(sentences[start_idx:end_idx])
            buffered.append(combined)
        return buffered

    def _calculate_cosine_distances(self, embeddings: list[list[float]]) -> list[float]:
        """Calculates cosine distance (1 - cosine_similarity) between consecutive vectors."""
        distances = []
        for i in range(len(embeddings) - 1):
            vec1 = np.array(embeddings[i], dtype=float)
            vec2 = np.array(embeddings[i + 1], dtype=float)

            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)

            if norm1 == 0.0 or norm2 == 0.0:
                sim = 0.0
            else:
                sim = float(np.dot(vec1, vec2) / (norm1 * norm2))
                sim = max(-1.0, min(1.0, sim))

            dist = 1.0 - sim
            distances.append(dist)
        return distances

    def _calculate_threshold(self, distances: list[float]) -> float:
        """Determines the distance cutoff threshold for segmenting text."""
        arr = np.array(distances, dtype=float)

        if self.breakpoint_threshold_type == "percentile":
            p = max(0.0, min(100.0, self.breakpoint_threshold_amount))
            return float(np.percentile(arr, p))

        elif self.breakpoint_threshold_type == "standard_deviation":
            mean_val = float(np.mean(arr))
            std_val = float(np.std(arr))
            return float(mean_val + self.breakpoint_threshold_amount * std_val)

        elif self.breakpoint_threshold_type == "interquartile":
            q75, q25 = np.percentile(arr, [75, 25])
            iqr = q75 - q25
            return float(q75 + self.breakpoint_threshold_amount * iqr)

        elif self.breakpoint_threshold_type == "gradient":
            if len(arr) < 2:
                return float(np.mean(arr))
            gradients = np.abs(np.diff(arr))
            grad_threshold = np.percentile(gradients, max(0.0, min(100.0, self.breakpoint_threshold_amount)))
            return float(np.mean(arr) + grad_threshold)

        elif self.breakpoint_threshold_type == "fixed":
            return float(self.breakpoint_threshold_amount)

        return float(np.percentile(arr, 90.0))

    def _identify_breakpoints(self, distances: list[float], threshold: float) -> list[int]:
        """Returns the 1-indexed sentence indices where a new chunk should start."""
        breakpoints = []
        for i, dist in enumerate(distances):
            if dist > threshold:
                breakpoints.append(i + 1)
        return breakpoints

    def _group_sentences_into_chunks(self, sentences: list[str], breakpoint_indices: list[int]) -> list[str]:
        """Slices sentences according to breakpoint indices and joins them into chunks."""
        chunks = []
        start_idx = 0
        bp_set = set(breakpoint_indices)

        for i in range(1, len(sentences)):
            if i in bp_set:
                chunk_text = " ".join(sentences[start_idx:i]).strip()
                if chunk_text:
                    chunks.append(chunk_text)
                start_idx = i

        last_chunk = " ".join(sentences[start_idx:]).strip()
        if last_chunk:
            chunks.append(last_chunk)

        return chunks

    def _enforce_size_constraints(self, chunks: list[str]) -> list[str]:
        """Merges undersized chunks and decomposes oversized chunks."""
        if not chunks:
            return []

        # Step A: Merge chunks smaller than min_chunk_size with neighbor
        merged_chunks = []
        buffer_text = ""

        for chunk in chunks:
            if not buffer_text:
                buffer_text = chunk
            elif len(buffer_text) < self.min_chunk_size:
                buffer_text = f"{buffer_text} {chunk}"
            else:
                merged_chunks.append(buffer_text)
                buffer_text = chunk

        if buffer_text:
            if merged_chunks and len(buffer_text) < self.min_chunk_size:
                merged_chunks[-1] = f"{merged_chunks[-1]} {buffer_text}"
            else:
                merged_chunks.append(buffer_text)

        # Step B: Split any chunks exceeding max_chunk_size using recursive fallback
        final_chunks = []
        recursive_fallback = RecursiveTextChunker(
            chunk_size=self.max_chunk_size,
            chunk_overlap=min(100, self.max_chunk_size // 10),
        )

        for chunk in merged_chunks:
            if len(chunk) > self.max_chunk_size:
                sub_chunks = recursive_fallback.split_text(chunk)
                final_chunks.extend(sub_chunks)
            else:
                final_chunks.append(chunk)

        return final_chunks

    def _fallback_sentence_chunking(self, sentences: list[str]) -> list[str]:
        """Simple sentence packing fallback when embeddings are not available."""
        chunks = []
        current = []
        curr_len = 0

        for s in sentences:
            if current and curr_len + len(s) > 500:
                chunks.append(" ".join(current))
                current = [s]
                curr_len = len(s)
            else:
                current.append(s)
                curr_len += len(s) + 1

        if current:
            chunks.append(" ".join(current))
        return chunks


def get_chunker(strategy: str = "recursive", **kwargs) -> BaseChunker:
    """Factory method to instantiate a text chunker based on strategy name.

    Supported strategies:
        - 'recursive': RecursiveTextChunker
        - 'semantic': SemanticChunker

    Args:
        strategy: 'recursive' or 'semantic'
        **kwargs: Strategy-specific parameter overrides.

    Returns:
        An instance of BaseChunker.
    """
    strat = (strategy or config.get("pipeline.chunking_strategy", "recursive")).lower().strip()

    if strat == "recursive":
        chunk_size = kwargs.get("chunk_size", config.get("pipeline.chunk_size", 500))
        chunk_overlap = kwargs.get("chunk_overlap", config.get("pipeline.chunk_overlap", 50))
        return RecursiveTextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    elif strat == "semantic":
        embedding_service = kwargs.get("embedding_service")
        if embedding_service is None:
            try:
                from src.embeddings.vertex import VertexEmbeddingService
                from google import genai

                gcp_project = config.get("gcp.project")
                gcp_location = config.get("gcp.location")
                model_name = config.get("models.embedding", "text-embedding-005")
                ai_client = genai.Client(vertexai=True, project=gcp_project, location=gcp_location)
                embedding_service = VertexEmbeddingService(client=ai_client, model_name=model_name)
            except Exception as e:
                logger.warning(f"Could not initialize live VertexEmbeddingService for SemanticChunker: {e}")

        threshold_type = kwargs.get(
            "breakpoint_threshold_type",
            config.get("pipeline.semantic_threshold_type", "percentile"),
        )
        threshold_amount = kwargs.get(
            "breakpoint_threshold_amount",
            config.get("pipeline.semantic_threshold_amount", 90.0),
        )
        buffer_size = kwargs.get(
            "buffer_size",
            config.get("pipeline.semantic_buffer_size", 1),
        )
        min_chunk_size = kwargs.get(
            "min_chunk_size",
            config.get("pipeline.semantic_min_chunk_size", 50),
        )
        max_chunk_size = kwargs.get(
            "max_chunk_size",
            config.get("pipeline.semantic_max_chunk_size", 2000),
        )

        return SemanticChunker(
            embedding_service=embedding_service,
            breakpoint_threshold_type=threshold_type,
            breakpoint_threshold_amount=threshold_amount,
            buffer_size=buffer_size,
            min_chunk_size=min_chunk_size,
            max_chunk_size=max_chunk_size,
        )

    else:
        raise ValueError(
            f"Unknown chunking strategy '{strategy}'. Supported strategies: 'recursive', 'semantic'"
        )

