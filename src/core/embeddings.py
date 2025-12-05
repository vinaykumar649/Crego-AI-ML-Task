"""
Embeddings module for generating and managing text embeddings.

Supports multiple embedding providers (sentence-transformers, OpenAI).
"""

import logging
from abc import ABC, abstractmethod
from typing import List

import numpy as np
from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)


class EmbeddingConfig(BaseSettings):
    """Configuration for embeddings."""

    provider: str = "sentence-transformers"
    model: str = "all-MiniLM-L6-v2"
    dimension: int = 384
    openai_api_key: str = ""

    class Config:
        env_prefix = "EMBEDDING_"


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""

    @abstractmethod
    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed

        Returns:
            numpy array of shape (len(texts), embedding_dimension)
        """
        pass

    @abstractmethod
    def embed_single(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.

        Args:
            text: Text string to embed

        Returns:
            numpy array of shape (embedding_dimension,)
        """
        pass


class SentenceTransformersProvider(EmbeddingProvider):
    """Embedding provider using sentence-transformers."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize sentence-transformers provider."""
        try:
            from sentence_transformers import SentenceTransformer

            logger.info(f"Loading sentence-transformers model: {model_name}")
            self.model = SentenceTransformer(model_name)
            self.dimension = self.model.get_sentence_embedding_dimension()
        except ImportError:
            raise ImportError("sentence-transformers is required. Install with: pip install sentence-transformers")

    def embed(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for multiple texts."""
        if not texts:
            return np.array([])
        logger.debug(f"Embedding {len(texts)} texts with sentence-transformers")
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings

    def embed_single(self, text: str) -> np.ndarray:
        """Generate embedding for a single text."""
        embeddings = self.embed([text])
        return embeddings[0]


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """Embedding provider using OpenAI API."""

    def __init__(self, api_key: str, model: str = "text-embedding-3-small"):
        """Initialize OpenAI embedding provider."""
        try:
            from openai import OpenAI

            self.client = OpenAI(api_key=api_key)
            self.model = model
            self.dimension = 1536 if model == "text-embedding-3-small" else 3072
            logger.info(f"Initialized OpenAI embeddings with model: {model}")
        except ImportError:
            raise ImportError("openai is required. Install with: pip install openai")

    def embed(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for multiple texts using OpenAI."""
        if not texts:
            return np.array([])
        logger.debug(f"Embedding {len(texts)} texts with OpenAI")
        response = self.client.embeddings.create(model=self.model, input=texts)
        embeddings = np.array([item.embedding for item in response.data])
        return embeddings

    def embed_single(self, text: str) -> np.ndarray:
        """Generate embedding for a single text using OpenAI."""
        embeddings = self.embed([text])
        return embeddings[0]


class EmbeddingManager:
    """Manages embedding generation with configurable provider."""

    def __init__(self, config: EmbeddingConfig):
        """Initialize embedding manager with configuration."""
        self.config = config
        self.provider = self._initialize_provider()

    def _initialize_provider(self) -> EmbeddingProvider:
        """Initialize the appropriate embedding provider."""
        if self.config.provider == "openai":
            if not self.config.openai_api_key:
                raise ValueError("EMBEDDING_OPENAI_API_KEY is required for OpenAI embeddings")
            return OpenAIEmbeddingProvider(api_key=self.config.openai_api_key, model=self.config.model)
        elif self.config.provider == "sentence-transformers":
            return SentenceTransformersProvider(model_name=self.config.model)
        else:
            raise ValueError(f"Unsupported embedding provider: {self.config.provider}")

    def embed(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts."""
        return self.provider.embed(texts)

    def embed_single(self, text: str) -> np.ndarray:
        """Generate embedding for a single text."""
        return self.provider.embed_single(text)

    def get_dimension(self) -> int:
        """Get embedding dimension."""
        return self.config.dimension
