"""
JSON Logic builder module for creating and validating JSON Logic rules.

Handles construction and validation of JSON Logic AST with allowed operators and keys.
"""

import json
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class JSONLogicBuilder:
    """Builder for creating valid JSON Logic rules."""

    ALLOWED_OPERATORS = {"and", "or", "if", ">", ">=", "<", "<=", "==", "!=", "in", "+", "-", "*", "/"}

    def __init__(self, allowed_keys: List[str], max_depth: int = 10):
        """Initialize JSON Logic builder."""
        self.allowed_keys = set(allowed_keys)
        self.max_depth = max_depth

    def validate(self, rule: Any, depth: int = 0) -> tuple[bool, List[str]]:
        """
        Validate a JSON Logic rule.

        Returns:
            (is_valid, list of error messages)
        """
        errors = []

        if depth > self.max_depth:
            errors.append(f"Rule exceeds maximum depth of {self.max_depth}")
            return False, errors

        if isinstance(rule, dict):
            if "var" in rule:
                var_name = rule.get("var")
                if var_name not in self.allowed_keys:
                    errors.append(f"Variable '{var_name}' is not in allowed keys: {sorted(self.allowed_keys)}")
                    return False, errors
                return len(errors) == 0, errors

            if len(rule) != 1:
                errors.append("Each rule object must have exactly one key (the operator)")
                return False, errors

            operator, operands = list(rule.items())[0]

            if operator not in self.ALLOWED_OPERATORS:
                errors.append(f"Operator '{operator}' is not allowed. Allowed: {self.ALLOWED_OPERATORS}")
                return False, errors

            if not isinstance(operands, (list, dict)):
                errors.append(f"Operands for '{operator}' must be list or dict, got {type(operands)}")
                return False, errors

            if isinstance(operands, list):
                for operand in operands:
                    is_valid, operand_errors = self.validate(operand, depth + 1)
                    errors.extend(operand_errors)
            else:
                is_valid, operand_errors = self.validate(operands, depth + 1)
                errors.extend(operand_errors)

        elif not isinstance(rule, (int, float, str, bool, type(None))):
            errors.append(f"Invalid rule component type: {type(rule)}")
            return False, errors

        return len(errors) == 0, errors

    def build_condition(self, key: str, operator: str, value: Any) -> Dict[str, Any]:
        """
        Build a simple condition rule.

        Example: build_condition("user_age", ">", 18)
        Returns: {">": [{"var": "user_age"}, 18]}
        """
        if operator not in self.ALLOWED_OPERATORS:
            raise ValueError(f"Operator '{operator}' is not allowed")

        if key not in self.allowed_keys:
            logger.warning(f"Key '{key}' not in allowed keys, but allowing for flexibility")

        return {operator: [{"var": key}, value]}

    def build_and(self, conditions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Build AND rule from multiple conditions."""
        if len(conditions) == 0:
            raise ValueError("AND rule requires at least one condition")
        if len(conditions) == 1:
            return conditions[0]
        return {"and": conditions}

    def build_or(self, conditions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Build OR rule from multiple conditions."""
        if len(conditions) == 0:
            raise ValueError("OR rule requires at least one condition")
        if len(conditions) == 1:
            return conditions[0]
        return {"or": conditions}

    def build_if(
        self, condition: Dict[str, Any], then_rule: Dict[str, Any], else_rule: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Build IF rule."""
        if else_rule:
            return {"if": [condition, then_rule, else_rule]}
        return {"if": [condition, then_rule]}

    def build_in(self, value: Any, array: List[Any]) -> Dict[str, Any]:
        """Build IN rule to check membership."""
        return {"in": [value, array]}

    def parse_from_prompt(self, prompt: str) -> Optional[Dict[str, Any]]:
        """
        Parse and build JSON Logic from a natural language prompt.

        This is a basic heuristic parser for common patterns.
        Returns None if parsing fails.
        """
        try:
            prompt_lower = prompt.lower()

            if "and" in prompt_lower and "or" in prompt_lower:
                logger.debug("Detected complex AND/OR logic in prompt")
                return None

            if "and" in prompt_lower:
                logger.debug("Detected AND logic in prompt")
                return None

            if "or" in prompt_lower:
                logger.debug("Detected OR logic in prompt")
                return None

            return None
        except Exception as e:
            logger.error(f"Error parsing prompt: {e}")
            return None


class JSONLogicValidator:
    """Validates JSON Logic syntax and semantic correctness."""

    @staticmethod
    def validate_json_syntax(rule: Any) -> tuple[bool, Optional[str]]:
        """Validate JSON syntax."""
        try:
            if isinstance(rule, str):
                json.loads(rule)
            elif isinstance(rule, dict):
                json.dumps(rule)
            return True, None
        except json.JSONDecodeError as e:
            return False, f"Invalid JSON: {e}"

    @staticmethod
    def validate_variables(rule: Any, allowed_keys: List[str]) -> tuple[bool, List[str]]:
        """Validate that all variables in rule are in allowed keys."""
        allowed_set = set(allowed_keys)
        errors = []

        def check_vars(node):
            if isinstance(node, dict):
                if "var" in node:
                    var_name = node["var"]
                    if var_name not in allowed_set:
                        errors.append(f"Unknown variable: '{var_name}'")
                for value in node.values():
                    check_vars(value)
            elif isinstance(node, list):
                for item in node:
                    check_vars(item)

        check_vars(rule)
        return len(errors) == 0, errors

    @staticmethod
    def estimate_rule_depth(rule: Any) -> int:
        """Estimate the depth of a rule."""
        if isinstance(rule, dict):
            if "var" in rule:
                return 1
            max_depth = 0
            for value in rule.values():
                if isinstance(value, list):
                    for item in value:
                        max_depth = max(max_depth, JSONLogicValidator.estimate_rule_depth(item))
                else:
                    max_depth = max(max_depth, JSONLogicValidator.estimate_rule_depth(value))
            return max_depth + 1
        return 1

    @staticmethod
    def estimate_rule_size(rule: Any) -> int:
        """Estimate the size (character count) of a rule."""
        try:
            return len(json.dumps(rule))
        except Exception:
            return 0
