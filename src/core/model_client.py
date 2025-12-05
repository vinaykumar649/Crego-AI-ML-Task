"""
Model client module for LLM interactions.

Supports multiple LLM providers (OpenAI, local models).
"""

import logging
from abc import ABC, abstractmethod

from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)


class LLMConfig(BaseSettings):
    """Configuration for LLM."""

    provider: str = "openai"
    model: str = "gpt-4-turbo"
    api_key: str = ""
    temperature: float = 0.7
    max_tokens: int = 2000
    timeout: int = 30

    class Config:
        env_prefix = "OPENAI_"


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def generate(self, prompt: str, system_prompt: str = "") -> str:
        """Generate response from LLM."""
        pass


class OpenAIProvider(LLMProvider):
    """OpenAI LLM provider."""

    def __init__(self, config: LLMConfig):
        """Initialize OpenAI provider."""
        try:
            from openai import OpenAI

            self.client = OpenAI(api_key=config.api_key)
            self.config = config
            logger.info(f"Initialized OpenAI provider with model: {config.model}")
        except ImportError:
            raise ImportError("openai is required. Install with: pip install openai")

    def generate(self, prompt: str, system_prompt: str = "") -> str:
        """Generate response using OpenAI API."""
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        try:
            logger.debug(f"Calling OpenAI with model {self.config.model}")
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                timeout=self.config.timeout,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error calling OpenAI: {e}")
            raise


class MockLLMProvider(LLMProvider):
    """Mock LLM provider for testing."""

    def generate(self, prompt: str, system_prompt: str = "") -> str:
        """Generate mock response."""
        logger.debug("Using mock LLM provider")
        return "This is a mock response."


class LLMClient:
    """Client for LLM interactions."""

    def __init__(self, config: LLMConfig):
        """Initialize LLM client."""
        self.config = config
        self.provider = self._initialize_provider()

    def _initialize_provider(self) -> LLMProvider:
        """Initialize the appropriate LLM provider."""
        if self.config.provider == "openai":
            if not self.config.api_key:
                logger.warning("OPENAI_API_KEY not set, using mock provider")
                return MockLLMProvider()
            return OpenAIProvider(self.config)
        elif self.config.provider == "mock":
            return MockLLMProvider()
        else:
            logger.warning(f"Unknown provider {self.config.provider}, using mock")
            return MockLLMProvider()

    def generate(self, prompt: str, system_prompt: str = "") -> str:
        """Generate response from LLM."""
        return self.provider.generate(prompt, system_prompt)

    def generate_json_logic_rule(self, user_prompt: str, store_keys: list, rag_context: str = "") -> tuple:
        """
        Generate JSON Logic rule from natural language prompt.

        Returns:
            (json_logic_dict, explanation)
        """
        system_prompt = self._build_system_prompt(store_keys, rag_context)
        response = self.generate(user_prompt, system_prompt)
        return response

    def _build_system_prompt(self, store_keys: list, rag_context: str = "") -> str:
        """Build system prompt for JSON Logic generation."""
        keys_str = ", ".join([f'"{key}"' for key in store_keys])

        system_prompt = f"""You are an expert at converting natural language business rules into JSON Logic format.

JSON Logic is a format for representing logical rules. You must use ONLY these operators:
- Logical: and, or, if
- Comparison: >, >=, <, <=, ==, !=
- Membership: in
- Arithmetic: +, -, *, /

Available data keys: {keys_str}

All variable references must use the format: {{"var": "<KEY_NAME>"}}

Rules for generating JSON Logic:
1. Parse the user's business rule carefully
2. Identify the conditions and operators
3. Build a valid JSON Logic AST
4. Only use the allowed operators and keys listed above
5. Ensure proper nesting and syntax
6. Provide a brief 1-3 sentence explanation

Output format:
{{
  "json_logic": {{ ... your JSON Logic rule ... }},
  "explanation": "Brief explanation of the rule in plain English"
}}

{f'Policy Context (for reference):{rag_context}' if rag_context else ''}
"""
        return system_prompt
