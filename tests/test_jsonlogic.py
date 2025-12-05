"""Tests for JSON Logic building and validation."""

import pytest

from src.core.jsonlogic_builder import JSONLogicBuilder, JSONLogicValidator
from src.core.validator import RuleValidator


class TestJSONLogicBuilder:
    """Tests for JSONLogicBuilder."""

    @pytest.fixture
    def builder(self):
        """Create a builder for testing."""
        allowed_keys = ["user_age", "purchase_amount", "is_premium_member"]
        return JSONLogicBuilder(allowed_keys)

    def test_build_simple_condition(self, builder):
        """Test building a simple condition."""
        rule = builder.build_condition("user_age", ">", 18)
        assert ">" in rule
        assert rule[">"][0] == {"var": "user_age"}
        assert rule[">"][1] == 18

    def test_build_and_rule(self, builder):
        """Test building an AND rule."""
        cond1 = builder.build_condition("user_age", ">", 18)
        cond2 = builder.build_condition("is_premium_member", "==", True)
        rule = builder.build_and([cond1, cond2])
        assert "and" in rule

    def test_build_or_rule(self, builder):
        """Test building an OR rule."""
        cond1 = builder.build_condition("user_age", ">", 18)
        cond2 = builder.build_condition("is_premium_member", "==", True)
        rule = builder.build_or([cond1, cond2])
        assert "or" in rule

    def test_build_if_rule(self, builder):
        """Test building an IF rule."""
        condition = builder.build_condition("is_premium_member", "==", True)
        then_rule = builder.build_condition("purchase_amount", ">", 100)
        rule = builder.build_if(condition, then_rule)
        assert "if" in rule

    def test_build_in_rule(self, builder):
        """Test building an IN rule."""
        rule = builder.build_in({"var": "user_age"}, [18, 21, 25])
        assert "in" in rule

    def test_validate_valid_rule(self, builder):
        """Test validation of a valid rule."""
        rule = builder.build_condition("user_age", ">", 18)
        is_valid, errors = builder.validate(rule)
        assert is_valid

    def test_validate_invalid_operator(self, builder):
        """Test validation with invalid operator."""
        rule = {"invalid_op": [{"var": "user_age"}, 18]}
        is_valid, errors = builder.validate(rule)
        assert not is_valid
        assert len(errors) > 0

    def test_validate_unknown_key(self, builder):
        """Test validation with unknown key."""
        rule = {">": [{"var": "unknown_key"}, 18]}
        is_valid, errors = builder.validate(rule)
        assert not is_valid


class TestJSONLogicValidator:
    """Tests for JSONLogicValidator."""

    def test_validate_json_syntax_valid(self):
        """Test JSON syntax validation with valid JSON."""
        rule = {"var": "user_age"}
        is_valid, error = JSONLogicValidator.validate_json_syntax(rule)
        assert is_valid
        assert error is None

    def test_validate_json_syntax_invalid(self):
        """Test JSON syntax validation with invalid JSON."""
        is_valid, error = JSONLogicValidator.validate_json_syntax("invalid json")
        assert not is_valid

    def test_validate_variables_valid(self):
        """Test variable validation with valid variables."""
        rule = {">": [{"var": "user_age"}, 18]}
        is_valid, errors = JSONLogicValidator.validate_variables(
            rule,
            ["user_age", "purchase_amount"]
        )
        assert is_valid
        assert len(errors) == 0

    def test_validate_variables_invalid(self):
        """Test variable validation with invalid variables."""
        rule = {">": [{"var": "unknown_var"}, 18]}
        is_valid, errors = JSONLogicValidator.validate_variables(
            rule,
            ["user_age", "purchase_amount"]
        )
        assert not is_valid
        assert len(errors) > 0

    def test_estimate_rule_depth(self):
        """Test rule depth estimation."""
        simple_rule = {"var": "user_age"}
        depth = JSONLogicValidator.estimate_rule_depth(simple_rule)
        assert depth >= 1

        nested_rule = {
            "and": [
                {">": [{"var": "user_age"}, 18]},
                {"==": [{"var": "is_premium_member"}, True]}
            ]
        }
        depth = JSONLogicValidator.estimate_rule_depth(nested_rule)
        assert depth > 1

    def test_estimate_rule_size(self):
        """Test rule size estimation."""
        rule = {"var": "user_age"}
        size = JSONLogicValidator.estimate_rule_size(rule)
        assert size > 0


class TestRuleValidator:
    """Tests for comprehensive RuleValidator."""

    @pytest.fixture
    def validator(self):
        """Create a validator for testing."""
        allowed_keys = ["user_age", "purchase_amount", "is_premium_member", "user_status"]
        return RuleValidator(allowed_keys)

    def test_validate_complete_valid_rule(self, validator):
        """Test complete validation of a valid rule."""
        rule = {">": [{"var": "user_age"}, 18]}
        is_valid, errors = validator.validate_complete(rule)
        assert is_valid
        assert len(errors) == 0

    def test_validate_complete_invalid_operator(self, validator):
        """Test complete validation with invalid operator."""
        rule = {"invalid_op": [{"var": "user_age"}, 18]}
        is_valid, errors = validator.validate_complete(rule)
        assert not is_valid

    def test_validate_complete_invalid_variable(self, validator):
        """Test complete validation with invalid variable."""
        rule = {">": [{"var": "unknown_var"}, 18]}
        is_valid, errors = validator.validate_complete(rule)
        assert not is_valid

    def test_validate_complete_deep_nesting(self, validator):
        """Test complete validation with deep nesting."""
        rule = {"var": "user_age"}
        for i in range(15):
            rule = {"and": [rule]}
        is_valid, errors = validator.validate_complete(rule)
        assert not is_valid

    def test_validate_complete_complex_rule(self, validator):
        """Test complete validation of a complex rule."""
        rule = {
            "and": [
                {">": [{"var": "user_age"}, 18]},
                {"==": [{"var": "is_premium_member"}, True]},
                {">=": [{"var": "purchase_amount"}, 100]}
            ]
        }
        is_valid, errors = validator.validate_complete(rule)
        assert is_valid
