"""Pytest configuration and shared fixtures."""

import json
import os
from pathlib import Path

import pytest

from src.core.embeddings import EmbeddingConfig, EmbeddingManager
from src.core.jsonlogic_builder import JSONLogicBuilder
from src.core.mapper import KeyMapper
from src.core.model_client import LLMConfig, LLMClient
from src.core.rag import RAGSystem
from src.core.validator import RuleValidator
from src.core.vector_store import InMemoryVectorStore


@pytest.fixture
def store_keys():
    """Load store keys from test data."""
    keys_file = Path(__file__).parent.parent / "data" / "sample_store_keys.json"
    if keys_file.exists():
        with open(keys_file, "r") as f:
            data = json.load(f)
        return [item["key"] for item in data.get("keys", [])]
    return [
        "user_age", "user_status", "purchase_amount", "is_premium_member",
        "account_age_days", "user_tags", "last_purchase_days_ago", "return_rate",
        "email_verified", "total_orders", "is_business_account", "region",
        "average_order_value", "loyalty_points", "subscription_active"
    ]


@pytest.fixture
def embedding_manager():
    """Create embedding manager for testing."""
    config = EmbeddingConfig(
        provider="sentence-transformers",
        model="all-MiniLM-L6-v2"
    )
    return EmbeddingManager(config)


@pytest.fixture
def vector_store():
    """Create in-memory vector store for testing."""
    return InMemoryVectorStore()


@pytest.fixture
def key_mapper(embedding_manager, vector_store, store_keys):
    """Create key mapper for testing."""
    return KeyMapper(
        embedding_manager,
        vector_store,
        store_keys,
        similarity_threshold=0.5,
        top_k=3
    )


@pytest.fixture
def llm_client():
    """Create mock LLM client for testing."""
    config = LLMConfig(
        provider="mock",
        api_key="test-key"
    )
    return LLMClient(config)


@pytest.fixture
def rag_system(embedding_manager, vector_store):
    """Create RAG system for testing."""
    policy_file = Path(__file__).parent.parent / "data" / "policy_docs.md"
    return RAGSystem(
        embedding_manager,
        vector_store,
        str(policy_file),
        top_k=3
    )


@pytest.fixture
def rule_validator(store_keys):
    """Create rule validator for testing."""
    return RuleValidator(store_keys)


@pytest.fixture
def json_logic_builder(store_keys):
    """Create JSON Logic builder for testing."""
    return JSONLogicBuilder(store_keys)
