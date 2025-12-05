"""
API routes for the JSON Logic Rule Generator.

Provides the single POST /generate-rule endpoint.
"""

import json
import logging
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class GenerateRuleRequest(BaseModel):
    """Request model for rule generation."""

    prompt: str = Field(..., description="Natural language prompt for rule generation")
    context_docs: Optional[List[str]] = Field(default=None, description="Optional list of context documents for RAG")


class KeyMapping(BaseModel):
    """Key mapping with similarity score."""

    user_phrase: str
    mapped_to: str
    similarity: float


class GenerateRuleResponse(BaseModel):
    """Response model for rule generation."""

    json_logic: dict
    explanation: str
    used_keys: List[str]
    key_mappings: List[KeyMapping]
    confidence_score: float


def create_routes(app_context):
    """Create API routes with application context."""
    router = APIRouter()

    @router.post("/generate-rule", response_model=GenerateRuleResponse)
    async def generate_rule(request: GenerateRuleRequest) -> GenerateRuleResponse:
        """
        Generate a JSON Logic rule from a natural language prompt.

        Args:
            request: Request containing prompt and optional context docs

        Returns:
            GenerateRuleResponse with JSON Logic, explanation, key mappings, and confidence score
        """
        try:
            logger.info(f"Generating rule for prompt: {request.prompt[:100]}...")

            mapper = app_context["mapper"]
            rag_system = app_context["rag_system"]
            llm_client = app_context["llm_client"]
            validator = app_context["validator"]
            store_keys = app_context["store_keys"]

            mappings, mapping_errors = mapper.map_phrases(request.prompt)

            if mapping_errors and not mappings:
                if len(mapping_errors) <= 3:
                    error_suggestions = ". ".join(mapping_errors)
                else:
                    error_suggestions = ". ".join(mapping_errors[:3]) + "..."
                raise HTTPException(
                    status_code=400,
                    detail=f"Phrase mapping confidence below threshold. Suggestions: {error_suggestions}",
                )

            used_keys = list(set([m.mapped_to for m in mappings]))

            rag_context = ""
            if app_context.get("config", {}).get("rag", {}).get("enabled", True):
                rag_context = rag_system.retrieve(request.prompt)

            system_prompt = _build_system_prompt(store_keys, rag_context, request.context_docs or [])

            llm_response = llm_client.generate(request.prompt, system_prompt)

            try:
                response_data = json.loads(llm_response)
                if "json_logic" not in response_data:
                    response_data = {"json_logic": llm_response, "explanation": "Generated rule"}
            except json.JSONDecodeError:
                response_data = {"json_logic": {}, "explanation": llm_response}

            json_logic = response_data.get("json_logic", {})
            explanation = response_data.get("explanation", "")

            is_valid, validation_errors = validator.validate_complete(json_logic)
            if not is_valid:
                logger.warning(f"Validation errors: {validation_errors}")

            mapping_similarities = [m.similarity for m in mappings]
            confidence_score = sum(mapping_similarities) / len(mapping_similarities) if mapping_similarities else 0.5

            key_mapping_dicts = [m.to_dict() for m in mappings]

            result = GenerateRuleResponse(
                json_logic=json_logic,
                explanation=explanation,
                used_keys=used_keys,
                key_mappings=key_mapping_dicts,
                confidence_score=round(confidence_score, 4),
            )

            logger.info(f"Successfully generated rule with confidence: {result.confidence_score}")
            return result

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error generating rule: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Error generating rule: {str(e)}")

    @router.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {"status": "ok"}

    @router.get("/keys")
    async def get_available_keys():
        """Get list of available store keys."""
        store_keys = app_context["store_keys"]
        return {"keys": store_keys, "count": len(store_keys)}

    return router


def _build_system_prompt(store_keys: List[str], rag_context: str = "", context_docs: List[str] = None) -> str:
    """Build system prompt for rule generation."""
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

Output must be valid JSON with this structure:
{{
  "json_logic": {{ ... your JSON Logic rule ... }},
  "explanation": "Brief 1-3 sentence explanation in plain English"
}}

Example:
User: "Apply discount if user is premium member AND purchase amount is greater than 100"
{{
  "json_logic": {{
    "and": [
      {{"var": "is_premium_member"}},
      {{">": [{{"var": "purchase_amount"}}, 100]}}
    ]
  }},
  "explanation": (
    "This rule applies a discount when the user is a premium member "
    "and their purchase amount exceeds $100."
  )
}}
"""

    if rag_context:
        system_prompt += f"\n\nRelevant Policy Context (use this to inform your rule generation):\n{rag_context}"

    if context_docs:
        system_prompt += "\n\nAdditional Context Documents:\n" + "\n".join(context_docs)

    return system_prompt
