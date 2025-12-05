"""
Validator module for comprehensive validation of generated rules.

Validates JSON Logic syntax, allowed keys, and semantic correctness.
"""

import json
import logging
from typing import Any, List, Optional, Tuple

logger = logging.getLogger(__name__)


class RuleValidator:
    """Comprehensive validator for JSON Logic rules."""

    def __init__(self, allowed_keys: List[str], config: dict = None):
        """Initialize rule validator."""
        self.allowed_keys = set(allowed_keys)
        self.config = config or {}
        self.max_depth = self.config.get("max_depth", 10)
        self.max_size = self.config.get("max_size", 5000)
        self.strict_mode = self.config.get("enable_strict_mode", True)

    def validate_complete(self, rule: Any) -> Tuple[bool, List[str]]:
        """
        Perform complete validation of a rule.

        Returns:
            (is_valid, list of error messages)
        """
        errors = []

        json_valid, json_error = self._validate_json_syntax(rule)
        if not json_valid:
            errors.append(json_error)
            return False, errors

        size_valid, size_error = self._validate_size(rule)
        if not size_valid:
            errors.append(size_error)

        depth_valid, depth_error = self._validate_depth(rule)
        if not depth_valid:
            errors.append(depth_error)

        vars_valid, var_errors = self._validate_variables(rule)
        if not vars_valid:
            errors.extend(var_errors)

        ops_valid, op_errors = self._validate_operators(rule)
        if not ops_valid:
            errors.extend(op_errors)

        return len(errors) == 0, errors

    def _validate_json_syntax(self, rule: Any) -> Tuple[bool, Optional[str]]:
        """Validate JSON syntax."""
        try:
            if isinstance(rule, str):
                json.loads(rule)
            else:
                json.dumps(rule)
            return True, None
        except (json.JSONDecodeError, TypeError) as e:
            return False, f"Invalid JSON syntax: {e}"

    def _validate_size(self, rule: Any) -> Tuple[bool, Optional[str]]:
        """Validate rule size."""
        try:
            rule_str = json.dumps(rule)
            size = len(rule_str)
            if size > self.max_size:
                return False, f"Rule size ({size} chars) exceeds maximum ({self.max_size} chars)"
            return True, None
        except Exception as e:
            return False, f"Error checking rule size: {e}"

    def _validate_depth(self, rule: Any, current_depth: int = 0) -> Tuple[bool, Optional[str]]:
        """Validate rule nesting depth."""
        if current_depth > self.max_depth:
            return False, f"Rule nesting depth exceeds maximum ({self.max_depth})"

        if isinstance(rule, dict):
            for key, value in rule.items():
                if key in ["and", "or", "if"]:
                    if isinstance(value, list):
                        for item in value:
                            valid, error = self._validate_depth(item, current_depth + 1)
                            if not valid:
                                return valid, error
                    else:
                        valid, error = self._validate_depth(value, current_depth + 1)
                        if not valid:
                            return valid, error
                elif key in [">", ">=", "<", "<=", "==", "!=", "in", "+", "-", "*", "/"]:
                    if isinstance(value, list):
                        for item in value:
                            valid, error = self._validate_depth(item, current_depth + 1)
                            if not valid:
                                return valid, error

        return True, None

    def _validate_variables(self, rule: Any) -> Tuple[bool, List[str]]:
        """Validate that all variables are in allowed keys."""
        errors = []

        def check_vars(node):
            if isinstance(node, dict):
                if "var" in node:
                    var_name = node.get("var")
                    if var_name not in self.allowed_keys:
                        errors.append(f"Unknown variable '{var_name}'. Allowed: {sorted(self.allowed_keys)}")
                else:
                    for key, value in node.items():
                        if key not in ["and", "or", "if", ">", ">=", "<", "<=", "==", "!=", "in", "+", "-", "*", "/"]:
                            logger.warning(f"Unknown operator: {key}")
                        if isinstance(value, list):
                            for item in value:
                                check_vars(item)
                        else:
                            check_vars(value)
            elif isinstance(node, list):
                for item in node:
                    check_vars(item)

        check_vars(rule)
        return len(errors) == 0, errors

    def _validate_operators(self, rule: Any) -> Tuple[bool, List[str]]:
        """Validate that only allowed operators are used."""
        allowed_operators = {"and", "or", "if", ">", ">=", "<", "<=", "==", "!=", "in", "+", "-", "*", "/"}
        errors = []

        def check_operators(node):
            if isinstance(node, dict):
                for key, value in node.items():
                    if key == "var":
                        continue
                    if key not in allowed_operators:
                        errors.append(f"Operator '{key}' is not allowed")
                    if isinstance(value, list):
                        for item in value:
                            check_operators(item)
                    else:
                        check_operators(value)
            elif isinstance(node, list):
                for item in node:
                    check_operators(item)

        check_operators(rule)
        return len(errors) == 0, errors


def validate_rule_complete(rule: Any, allowed_keys: List[str], config: dict = None) -> Tuple[bool, List[str]]:
    """
    Convenience function to validate a rule completely.

    Returns:
        (is_valid, list of error messages)
    """
    validator = RuleValidator(allowed_keys, config)
    return validator.validate_complete(rule)
