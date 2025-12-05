"""
Main FastAPI application for JSON Logic Rule Generator.

Initializes all components and serves the API.
"""

import json
import logging
import os
from pathlib import Path

import uvicorn
import yaml
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import create_routes
from src.core.embeddings import EmbeddingConfig, EmbeddingManager
from src.core.mapper import KeyMapper
from src.core.model_client import LLMConfig, LLMClient
from src.core.rag import RAGSystem
from src.core.validator import RuleValidator
from src.core.vector_store import create_vector_store

load_dotenv()

log_level = os.getenv("LOG_LEVEL", "INFO")
logging.basicConfig(
    level=getattr(logging, log_level.upper(), logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_config():
    """Load configuration from config.yaml with env variable substitution."""
    config_path = Path("configs/config.yaml")

    if not config_path.exists():
        logger.warning(f"Config file not found at {config_path}, using defaults")
        return {}

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    config = _substitute_env_vars(config)
    logger.info("Configuration loaded successfully")
    return config


def _substitute_env_vars(obj):
    """Recursively substitute environment variables in config."""
    if isinstance(obj, str):
        if obj.startswith("${") and obj.endswith("}"):
            var_spec = obj[2:-1]
            if ":" in var_spec:
                var_name, default = var_spec.split(":", 1)
                return os.getenv(var_name, default)
            else:
                return os.getenv(var_spec, "")
        return obj
    elif isinstance(obj, dict):
        return {k: _substitute_env_vars(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_substitute_env_vars(item) for item in obj]
    return obj


def load_store_keys(path: str = "data/sample_store_keys.json") -> list:
    """Load store keys from JSON file."""
    if not os.path.exists(path):
        logger.warning(f"Store keys file not found at {path}, using empty list")
        return []

    with open(path, "r") as f:
        data = json.load(f)

    keys = [item["key"] for item in data.get("keys", [])]
    logger.info(f"Loaded {len(keys)} store keys")
    return keys


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    logger.info("Starting JSON Logic Rule Generator")

    config = load_config()
    store_keys = load_store_keys()

    embedding_config = EmbeddingConfig(
        provider=os.getenv("EMBEDDING_PROVIDER", "sentence-transformers"),
        model=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
    )

    embedding_manager = EmbeddingManager(embedding_config)
    logger.info(f"Embeddings initialized with provider: {embedding_config.provider}")

    vector_store_type = os.getenv("VECTOR_STORE_TYPE", "in-memory")
    vector_store = create_vector_store(vector_store_type, embedding_manager.config.dimension)
    logger.info(f"Vector store initialized: {vector_store_type}")

    similarity_threshold = float(os.getenv("SIMILARITY_THRESHOLD", "0.75"))
    mapper = KeyMapper(
        embedding_manager,
        vector_store,
        store_keys,
        similarity_threshold=similarity_threshold,
        top_k=int(os.getenv("SIMILARITY_TOP_K", "3")),
    )
    logger.info(f"Key mapper initialized with threshold: {similarity_threshold}")

    llm_config = LLMConfig(
        provider=os.getenv("OPENAI_PROVIDER", "openai"),
        model=os.getenv("OPENAI_MODEL", "gpt-4-turbo"),
        api_key=os.getenv("OPENAI_API_KEY", ""),
        temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.7")),
        max_tokens=int(os.getenv("OPENAI_MAX_TOKENS", "2000")),
    )

    llm_client = LLMClient(llm_config)
    logger.info(f"LLM client initialized with model: {llm_config.model}")

    policy_docs_path = "data/policy_docs.md"
    rag_system = RAGSystem(embedding_manager, vector_store, policy_docs_path, top_k=int(os.getenv("RAG_TOP_K", "3")))
    logger.info("RAG system initialized")

    validator_config = config.get("jsonlogic", {})
    validator = RuleValidator(store_keys, validator_config)
    logger.info("Rule validator initialized")

    app = FastAPI(
        title="JSON Logic Rule Generator",
        description="Generate JSON Logic rules from natural language prompts with RAG and embeddings",
        version="1.0.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app_context = {
        "config": config,
        "store_keys": store_keys,
        "embedding_manager": embedding_manager,
        "vector_store": vector_store,
        "mapper": mapper,
        "llm_client": llm_client,
        "rag_system": rag_system,
        "validator": validator,
    }

    routes = create_routes(app_context)
    app.include_router(routes)

    logger.info("FastAPI application created successfully")
    return app


app = create_app()


if __name__ == "__main__":
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    workers = int(os.getenv("API_WORKERS", "4"))

    logger.info(f"Starting server on {host}:{port} with {workers} workers")
    uvicorn.run("src.main:app", host=host, port=port, workers=workers, reload=False)
