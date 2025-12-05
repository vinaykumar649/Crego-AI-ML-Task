"""
Vector store module for managing indexed embeddings.

Supports FAISS for efficient similarity search and in-memory cosine similarity.
"""

import logging
import os
from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class VectorStore(ABC):
    """Abstract base class for vector stores."""

    @abstractmethod
    def add_texts(self, texts: List[str], embeddings: np.ndarray, ids: List[str] = None) -> None:
        """Add texts and their embeddings to the store."""
        pass

    @abstractmethod
    def search(self, embedding: np.ndarray, top_k: int = 5) -> List[Tuple[str, float, int]]:
        """Search for similar embeddings. Returns list of (id, score, index)."""
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """Save the vector store to disk."""
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """Load the vector store from disk."""
        pass


class InMemoryVectorStore(VectorStore):
    """Simple in-memory vector store using cosine similarity."""

    def __init__(self):
        """Initialize in-memory vector store."""
        self.embeddings = np.array([])
        self.texts = []
        self.ids = []
        logger.info("Initialized in-memory vector store")

    def add_texts(self, texts: List[str], embeddings: np.ndarray, ids: List[str] = None) -> None:
        """Add texts and embeddings to the store."""
        if ids is None:
            ids = [str(i) for i in range(len(texts))]

        if self.embeddings.size == 0:
            self.embeddings = embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, embeddings])

        self.texts.extend(texts)
        self.ids.extend(ids)
        logger.debug(f"Added {len(texts)} texts to vector store. Total: {len(self.texts)}")

    def search(self, embedding: np.ndarray, top_k: int = 5) -> List[Tuple[str, float, int]]:
        """Search for similar embeddings using cosine similarity."""
        if self.embeddings.size == 0:
            return []

        embedding = embedding.reshape(1, -1)
        similarities = cosine_similarity(embedding, self.embeddings)[0]

        top_indices = np.argsort(similarities)[::-1][:top_k]
        results = [(self.ids[i], float(similarities[i]), i) for i in top_indices]
        return results

    def save(self, path: str) -> None:
        """Save embeddings and metadata to disk."""
        os.makedirs(path, exist_ok=True)
        np.save(os.path.join(path, "embeddings.npy"), self.embeddings)

        with open(os.path.join(path, "metadata.txt"), "w") as f:
            for text_id, text in zip(self.ids, self.texts):
                f.write(f"{text_id}\t{text}\n")
        logger.info(f"Saved vector store to {path}")

    def load(self, path: str) -> None:
        """Load embeddings and metadata from disk."""
        try:
            embeddings_path = os.path.join(path, "embeddings.npy")
            metadata_path = os.path.join(path, "metadata.txt")

            if os.path.exists(embeddings_path):
                self.embeddings = np.load(embeddings_path)
                logger.info(f"Loaded embeddings from {embeddings_path}")

            if os.path.exists(metadata_path):
                with open(metadata_path, "r") as f:
                    for line in f:
                        parts = line.strip().split("\t", 1)
                        if len(parts) == 2:
                            self.ids.append(parts[0])
                            self.texts.append(parts[1])
                logger.info(f"Loaded {len(self.ids)} texts from {metadata_path}")
        except Exception as e:
            logger.error(f"Error loading vector store from {path}: {e}")


class FAISSVectorStore(VectorStore):
    """Vector store using FAISS for efficient similarity search."""

    def __init__(self, dimension: int):
        """Initialize FAISS vector store."""
        try:
            import faiss

            self.faiss = faiss
            self.index = faiss.IndexFlatL2(dimension)
            self.texts = []
            self.ids = []
            self.dimension = dimension
            logger.info(f"Initialized FAISS vector store with dimension {dimension}")
        except ImportError:
            raise ImportError("faiss is required. Install with: pip install faiss-cpu")

    def add_texts(self, texts: List[str], embeddings: np.ndarray, ids: List[str] = None) -> None:
        """Add texts and embeddings to FAISS index."""
        if ids is None:
            ids = [str(i) for i in range(len(texts))]

        embeddings = np.ascontiguousarray(embeddings).astype("float32")
        self.index.add(embeddings)
        self.texts.extend(texts)
        self.ids.extend(ids)
        logger.debug(f"Added {len(texts)} texts to FAISS. Total: {self.index.ntotal}")

    def search(self, embedding: np.ndarray, top_k: int = 5) -> List[Tuple[str, float, int]]:
        """Search for similar embeddings using FAISS."""
        if self.index.ntotal == 0:
            return []

        embedding = np.ascontiguousarray(embedding).reshape(1, -1).astype("float32")
        distances, indices = self.index.search(embedding, min(top_k, self.index.ntotal))

        results = []
        for i, idx in enumerate(indices[0]):
            if idx >= 0 and idx < len(self.ids):
                similarity = 1 / (1 + distances[0][i])
                results.append((self.ids[idx], float(similarity), idx))

        return results

    def save(self, path: str) -> None:
        """Save FAISS index and metadata to disk."""
        os.makedirs(path, exist_ok=True)
        self.faiss.write_index(self.index, os.path.join(path, "index.faiss"))

        with open(os.path.join(path, "metadata.txt"), "w") as f:
            for text_id, text in zip(self.ids, self.texts):
                f.write(f"{text_id}\t{text}\n")
        logger.info(f"Saved FAISS vector store to {path}")

    def load(self, path: str) -> None:
        """Load FAISS index and metadata from disk."""
        try:
            index_path = os.path.join(path, "index.faiss")
            metadata_path = os.path.join(path, "metadata.txt")

            if os.path.exists(index_path):
                self.index = self.faiss.read_index(index_path)
                logger.info(f"Loaded FAISS index from {index_path}")

            if os.path.exists(metadata_path):
                with open(metadata_path, "r") as f:
                    for line in f:
                        parts = line.strip().split("\t", 1)
                        if len(parts) == 2:
                            self.ids.append(parts[0])
                            self.texts.append(parts[1])
                logger.info(f"Loaded {len(self.ids)} texts from {metadata_path}")
        except Exception as e:
            logger.error(f"Error loading FAISS vector store from {path}: {e}")


def create_vector_store(store_type: str, dimension: int = 384) -> VectorStore:
    """Factory function to create appropriate vector store."""
    if store_type == "faiss":
        return FAISSVectorStore(dimension)
    elif store_type == "in-memory":
        return InMemoryVectorStore()
    else:
        logger.warning(f"Unknown vector store type {store_type}, using in-memory")
        return InMemoryVectorStore()
