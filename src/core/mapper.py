"""
Mapper module for mapping user phrases to store keys using embeddings and similarity.
"""

import logging
import re
from typing import List, Optional

from src.core.embeddings import EmbeddingManager
from src.core.vector_store import VectorStore

logger = logging.getLogger(__name__)


class KeyMapping:
    """Represents a mapping from a user phrase to a store key."""

    def __init__(self, user_phrase: str, mapped_to: str, similarity: float):
        """Initialize key mapping."""
        self.user_phrase = user_phrase
        self.mapped_to = mapped_to
        self.similarity = similarity

    def to_dict(self):
        """Convert to dictionary."""
        return {"user_phrase": self.user_phrase, "mapped_to": self.mapped_to, "similarity": round(self.similarity, 4)}


class PhraseExtractor:
    """Extracts potential key phrases from natural language prompts."""

    @staticmethod
    def extract_phrases(text: str) -> List[str]:
        """
        Extract potential key phrases from text.

        This is a simple heuristic-based extraction that looks for:
        - Words in quotes
        - Capitalized terms
        - Numbers and percentages
        - Common store-related terms
        """
        phrases = []

        quoted = re.findall(r'"([^"]+)"', text)
        phrases.extend(quoted)

        words = text.split()
        for word in words:
            if len(word) > 3:
                clean_word = re.sub(r"[^\w]", "", word)
                if clean_word and clean_word[0].isupper():
                    phrases.append(clean_word)

        store_terms = [
            "age",
            "status",
            "purchase",
            "premium",
            "member",
            "account",
            "order",
            "return",
            "email",
            "tag",
            "business",
            "region",
            "loyalty",
            "subscription",
        ]
        for term in store_terms:
            if term.lower() in text.lower():
                phrases.append(term)

        phrases = list(set(phrases))
        return [p for p in phrases if len(p) > 0]

    @staticmethod
    def extract_numeric_values(text: str) -> List[tuple]:
        """
        Extract numeric values and ranges from text.

        Returns list of (value, unit) tuples.
        """
        values = []

        numbers = re.findall(r"\d+(?:\.\d+)?", text)
        for num in numbers:
            values.append((float(num), None))

        ranges = re.findall(r"(\d+(?:\.\d+)?)\s*(?:to|through|than|less|more)\s*(\d+(?:\.\d+)?)", text)
        for start, end in ranges:
            values.append((float(start), float(end)))

        return values

    @staticmethod
    def extract_booleans(text: str) -> List[tuple]:
        """Extract boolean indicators from text."""
        booleans = []
        text_lower = text.lower()

        if any(word in text_lower for word in ["is", "must be", "should be", "can be"]):
            booleans.append(("is", True))

        if any(word in text_lower for word in ["not", "cannot", "cannot be", "must not"]):
            booleans.append(("not", False))

        return booleans


class KeyMapper:
    """Maps user phrases to store keys using embeddings and similarity."""

    def __init__(
        self,
        embedding_manager: EmbeddingManager,
        vector_store: VectorStore,
        store_keys: List[str],
        similarity_threshold: float = 0.75,
        top_k: int = 3,
    ):
        """Initialize key mapper."""
        self.embedding_manager = embedding_manager
        self.vector_store = vector_store
        self.store_keys = store_keys
        self.similarity_threshold = similarity_threshold
        self.top_k = top_k
        self.phrase_extractor = PhraseExtractor()

        self._index_store_keys()

    def _index_store_keys(self) -> None:
        """Index store keys in the vector store."""
        logger.info(f"Indexing {len(self.store_keys)} store keys")
        embeddings = self.embedding_manager.embed(self.store_keys)
        self.vector_store.add_texts(self.store_keys, embeddings, ids=self.store_keys)

    def map_phrases(self, text: str) -> tuple[List[KeyMapping], List[str]]:
        """
        Map phrases from text to store keys.

        Returns:
            (mappings: List of KeyMapping objects, errors: List of error messages)
        """
        phrases = self.phrase_extractor.extract_phrases(text)
        logger.debug(f"Extracted {len(phrases)} phrases from text")

        mappings = []
        errors = []

        for phrase in phrases:
            phrase_embedding = self.embedding_manager.embed_single(phrase)
            results = self.vector_store.search(phrase_embedding, top_k=self.top_k)

            if not results:
                errors.append(f"No matches found for phrase: {phrase}")
                continue

            best_match_key, best_similarity, _ = results[0]

            if best_similarity < self.similarity_threshold:
                suggestions = [key for key, sim, _ in results[:3]]
                error_msg = (
                    f"Phrase '{phrase}' mapping confidence too low ({best_similarity:.2f}). "
                    f"Suggestions: {suggestions}"
                )
                errors.append(error_msg)
                logger.warning(error_msg)
            else:
                mapping = KeyMapping(phrase, best_match_key, best_similarity)
                mappings.append(mapping)
                logger.debug(f"Mapped '{phrase}' to '{best_match_key}' (similarity: {best_similarity:.4f})")

        return mappings, errors

    def get_all_potential_matches(self, phrase: str, top_k: Optional[int] = None) -> List[tuple]:
        """Get all potential matches for a phrase with similarity scores."""
        if top_k is None:
            top_k = self.top_k

        phrase_embedding = self.embedding_manager.embed_single(phrase)
        results = self.vector_store.search(phrase_embedding, top_k=top_k)
        return results
