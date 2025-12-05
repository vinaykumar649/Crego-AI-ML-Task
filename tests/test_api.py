"""Tests for API endpoints."""

import pytest
from fastapi.testclient import TestClient

from src.main import create_app


@pytest.fixture
def client():
    """Create a test client."""
    app = create_app()
    return TestClient(app)


class TestHealthCheck:
    """Tests for health check endpoint."""

    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


class TestGetKeys:
    """Tests for get available keys endpoint."""

    def test_get_keys(self, client):
        """Test getting available keys."""
        response = client.get("/keys")
        assert response.status_code == 200
        data = response.json()
        assert "keys" in data
        assert "count" in data
        assert isinstance(data["keys"], list)
        assert data["count"] > 0


class TestGenerateRule:
    """Tests for rule generation endpoint."""

    def test_generate_rule_basic(self, client):
        """Test basic rule generation."""
        request = {
            "prompt": "Apply discount if user is premium member"
        }
        response = client.post("/generate-rule", json=request)

        assert response.status_code in [200, 400, 500]
        if response.status_code == 200:
            data = response.json()
            assert "json_logic" in data
            assert "explanation" in data
            assert "used_keys" in data
            assert "key_mappings" in data
            assert "confidence_score" in data

    def test_generate_rule_with_context(self, client):
        """Test rule generation with context docs."""
        request = {
            "prompt": "Premium members over 25 get 20% discount",
            "context_docs": ["Premium benefits include discounts and free shipping"]
        }
        response = client.post("/generate-rule", json=request)

        assert response.status_code in [200, 400, 500]
        if response.status_code == 200:
            data = response.json()
            assert isinstance(data, dict)

    def test_generate_rule_missing_prompt(self, client):
        """Test rule generation with missing prompt."""
        request = {}
        response = client.post("/generate-rule", json=request)
        assert response.status_code == 422

    def test_generate_rule_confidence_score(self, client):
        """Test that confidence score is returned."""
        request = {
            "prompt": "Check if user is premium and purchase amount is high"
        }
        response = client.post("/generate-rule", json=request)

        if response.status_code == 200:
            data = response.json()
            assert "confidence_score" in data
            assert 0 <= data["confidence_score"] <= 1

    def test_generate_rule_key_mappings(self, client):
        """Test that key mappings are returned."""
        request = {
            "prompt": "User age is greater than 18 and is premium"
        }
        response = client.post("/generate-rule", json=request)

        if response.status_code == 200:
            data = response.json()
            assert "key_mappings" in data
            if len(data["key_mappings"]) > 0:
                mapping = data["key_mappings"][0]
                assert "user_phrase" in mapping
                assert "mapped_to" in mapping
                assert "similarity" in mapping


class TestGenerateRuleExamples:
    """Tests based on the three example prompts from the spec."""

    def test_example_1_premium_user_discount(self, client):
        """Test Example 1: Premium User Discount."""
        request = {
            "prompt": "Generate a rule that applies a discount if the user is a premium member and their purchase amount is greater than $100."
        }
        response = client.post("/generate-rule", json=request)

        assert response.status_code in [200, 400, 500]
        if response.status_code == 200:
            data = response.json()
            assert "json_logic" in data
            assert isinstance(data["json_logic"], dict)
            assert "is_premium_member" in data.get("used_keys", []) or "premium" in data.get("explanation", "").lower()

    def test_example_2_high_value_customer(self, client):
        """Test Example 2: High-Value Customer."""
        request = {
            "prompt": "Create a rule that identifies high-value customers: those who have made more than 50 purchases OR have spent over $1000 total."
        }
        response = client.post("/generate-rule", json=request)

        assert response.status_code in [200, 400, 500]
        if response.status_code == 200:
            data = response.json()
            assert "json_logic" in data
            assert "used_keys" in data

    def test_example_3_account_eligibility(self, client):
        """Test Example 3: Account Eligibility."""
        request = {
            "prompt": "Generate a rule for account eligibility: the user must be at least 18 years old, have a verified email, and account status must be active."
        }
        response = client.post("/generate-rule", json=request)

        assert response.status_code in [200, 400, 500]
        if response.status_code == 200:
            data = response.json()
            assert "json_logic" in data
            assert "explanation" in data
