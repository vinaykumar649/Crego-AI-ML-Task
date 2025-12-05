"""
RAG (Retrieval-Augmented Generation) module for policy document retrieval.

Provides context from policy documents for LLM prompt enrichment.
"""

import logging
import os
from typing import List, Tuple

from src.core.embeddings import EmbeddingManager
from src.core.vector_store import VectorStore

logger = logging.getLogger(__name__)


class RAGSystem:
    """Retrieval-Augmented Generation system for policy documents."""

    def __init__(
        self,
        embedding_manager: EmbeddingManager,
        vector_store: VectorStore,
        policy_docs_path: str,
        top_k: int = 3,
        similarity_threshold: float = 0.5,
    ):
        """Initialize RAG system."""
        self.embedding_manager = embedding_manager
        self.vector_store = vector_store
        self.policy_docs_path = policy_docs_path
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold

        self._load_and_index_policy_docs()

    def _load_and_index_policy_docs(self) -> None:
        """Load and index policy documents."""
        if not os.path.exists(self.policy_docs_path):
            logger.warning(f"Policy docs file not found: {self.policy_docs_path}")
            return

        try:
            with open(self.policy_docs_path, "r", encoding="utf-8") as f:
                content = f.read()

            chunks = self._chunk_policy_docs(content)
            logger.info(f"Loaded {len(chunks)} policy document chunks")

            embeddings = self.embedding_manager.embed([chunk[0] for chunk in chunks])
            ids = [f"policy_{i}" for i in range(len(chunks))]

            self.vector_store.add_texts([chunk[0] for chunk in chunks], embeddings, ids=ids)
            logger.info(f"Indexed {len(chunks)} policy document chunks")
        except Exception as e:
            logger.error(f"Error loading policy docs: {e}")

    def _chunk_policy_docs(self, content: str, chunk_size: int = 500) -> List[Tuple[str, str]]:
        """
        Split policy documents into chunks for indexing.

        Returns:
            List of (chunk_text, source) tuples
        """
        sections = content.split("##")
        chunks = []

        for section in sections:
            section = section.strip()
            if not section:
                continue

            lines = section.split("\n")
            section_title = lines[0] if lines else "Unknown"

            current_chunk = ""
            for line in lines[1:]:
                if len(current_chunk) + len(line) > chunk_size:
                    if current_chunk:
                        chunks.append((current_chunk.strip(), section_title))
                    current_chunk = line
                else:
                    current_chunk += "\n" + line

            if current_chunk:
                chunks.append((current_chunk.strip(), section_title))

        return chunks

    def retrieve(self, query: str) -> str:
        """
        Retrieve relevant policy snippets for a query.

        Returns:
            Formatted context string for inclusion in LLM prompt
        """
        if (
            self.vector_store.index.ntotal == 0
            if hasattr(self.vector_store, "index")
            else len(self.vector_store.texts) == 0
        ):
            return ""

        query_embedding = self.embedding_manager.embed_single(query)
        results = self.vector_store.search(query_embedding, top_k=self.top_k)

        context_parts = []
        for text_id, similarity, _ in results:
            if similarity >= self.similarity_threshold:
                context_parts.append(text_id)

        if not context_parts:
            return ""

        context = "\n\n".join([f"- {part}" for part in context_parts[: self.top_k]])
        return context

    def retrieve_with_scores(self, query: str) -> List[Tuple[str, float]]:
        """
        Retrieve relevant policy snippets with similarity scores.

        Returns:
            List of (text, similarity_score) tuples
        """
        if (
            self.vector_store.index.ntotal == 0
            if hasattr(self.vector_store, "index")
            else len(self.vector_store.texts) == 0
        ):
            return []

        query_embedding = self.embedding_manager.embed_single(query)
        results = self.vector_store.search(query_embedding, top_k=self.top_k)

        context_results = [
            (text_id, similarity) for text_id, similarity, _ in results if similarity >= self.similarity_threshold
        ]
        return context_results
