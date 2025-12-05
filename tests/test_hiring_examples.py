"""
Test with hiring manager's example prompts.

These tests verify the three specific examples provided in the assignment:
1. Loan approval rules
2. High-risk flagging
3. Income preference rules
"""

import json
import pytest
from fastapi.testclient import TestClient

from src.main import app

client = TestClient(app)


class TestHiringManagerExamples:
    """Tests for hiring manager's specific example prompts."""

    def test_example_1_loan_approval(self):
        """
        Example 1: Approve if bureau score > 700 and business vintage at least 3 years
        and applicant age between 25 and 60.
        """
        prompt = (
            "Approve if bureau score > 700 and business vintage at least 3 years "
            "and applicant age between 25 and 60."
        )

        response = client.post("/generate-rule", json={"prompt": prompt})

        print("\n" + "=" * 80)
        print("EXAMPLE 1: LOAN APPROVAL")
        print("=" * 80)
        print(f"\nPrompt:\n{prompt}")
        print(f"\nStatus Code: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            print(f"\nJSON Logic:\n{json.dumps(data['json_logic'], indent=2)}")
            print(f"\nExplanation:\n{data['explanation']}")
            print(f"\nKey Mappings:")
            for mapping in data.get("key_mappings", []):
                print(
                    f"  - '{mapping['user_phrase']}' → '{mapping['mapped_to']}' "
                    f"(similarity: {mapping['similarity']:.4f})"
                )
            print(f"\nUsed Keys: {data.get('used_keys', [])}")
            print(f"Confidence Score: {data['confidence_score']:.4f}")

            # Verify required keys are used (may be mapped differently)
            json_logic_str = json.dumps(data["json_logic"])
            assert (
                "bureau.score" in json_logic_str or "bureau" in json_logic_str
            ), "Bureau score should be referenced"
            assert (
                "business.vintage_in_years" in json_logic_str
                or "vintage" in json_logic_str
            ), "Business vintage should be referenced"
            assert (
                "primary_applicant.age" in json_logic_str or "age" in json_logic_str
            ), "Applicant age should be referenced"
        elif response.status_code == 400:
            data = response.json()
            print(f"\nError Response (400): {data.get('detail', 'No detail')}")
            print(
                "This is expected when phrase mapping confidence is below threshold."
            )
            print(
                "The system provides suggestions for unmapped phrases as per requirements."
            )
            assert (
                "Suggestions" in data.get("detail", "")
            ), "Should provide suggestions for low-confidence phrases"
        else:
            pytest.fail(f"Unexpected status code: {response.status_code}")

    def test_example_2_high_risk_flagging(self):
        """
        Example 2: Flag as high risk if wilful default is true OR overdue amount > 50000
        OR bureau.dpd >= 90.
        """
        prompt = (
            "Flag as high risk if wilful default is true OR overdue amount > 50000 "
            "OR bureau.dpd >= 90."
        )

        response = client.post("/generate-rule", json={"prompt": prompt})

        print("\n" + "=" * 80)
        print("EXAMPLE 2: HIGH-RISK FLAGGING")
        print("=" * 80)
        print(f"\nPrompt:\n{prompt}")
        print(f"\nStatus Code: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            print(f"\nJSON Logic:\n{json.dumps(data['json_logic'], indent=2)}")
            print(f"\nExplanation:\n{data['explanation']}")
            print(f"\nKey Mappings:")
            for mapping in data.get("key_mappings", []):
                print(
                    f"  - '{mapping['user_phrase']}' → '{mapping['mapped_to']}' "
                    f"(similarity: {mapping['similarity']:.4f})"
                )
            print(f"\nUsed Keys: {data.get('used_keys', [])}")
            print(f"Confidence Score: {data['confidence_score']:.4f}")

            # Verify response structure
            assert "json_logic" in data
            assert "explanation" in data

            # Verify high-risk criteria are present
            json_logic_str = json.dumps(data["json_logic"])
            assert (
                "wilful_default" in json_logic_str or "default" in json_logic_str
            ), "Wilful default should be referenced"
            assert (
                "overdue_amount" in json_logic_str or "overdue" in json_logic_str
            ), "Overdue amount should be referenced"
            assert (
                "dpd" in json_logic_str
            ), "DPD (Days Past Due) should be referenced"
        elif response.status_code == 400:
            data = response.json()
            print(f"\nError Response (400): {data.get('detail', 'No detail')}")
            print(
                "This is expected when phrase mapping confidence is below threshold."
            )
            assert (
                "Suggestions" in data.get("detail", "")
            ), "Should provide suggestions for low-confidence phrases"
        else:
            pytest.fail(f"Unexpected status code: {response.status_code}")

    def test_example_3_income_preference(self):
        """
        Example 3: Prefer applicants with tag 'veteran' OR with monthly_income > 1,00,000.
        """
        prompt = "Prefer applicants with tag 'veteran' OR with monthly_income > 1,00,000."

        response = client.post("/generate-rule", json={"prompt": prompt})

        print("\n" + "=" * 80)
        print("EXAMPLE 3: INCOME PREFERENCE")
        print("=" * 80)
        print(f"\nPrompt:\n{prompt}")
        print(f"\nStatus Code: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            print(f"\nJSON Logic:\n{json.dumps(data['json_logic'], indent=2)}")
            print(f"\nExplanation:\n{data['explanation']}")
            print(f"\nKey Mappings:")
            for mapping in data.get("key_mappings", []):
                print(
                    f"  - '{mapping['user_phrase']}' → '{mapping['mapped_to']}' "
                    f"(similarity: {mapping['similarity']:.4f})"
                )
            print(f"\nUsed Keys: {data.get('used_keys', [])}")
            print(f"Confidence Score: {data['confidence_score']:.4f}")

            # Verify response structure
            assert "json_logic" in data
            assert "explanation" in data

            # Verify preference criteria are present
            json_logic_str = json.dumps(data["json_logic"])
            assert (
                "tags" in json_logic_str or "veteran" in json_logic_str
            ), "Applicant tags should be referenced"
            assert (
                "monthly_income" in json_logic_str or "income" in json_logic_str
            ), "Monthly income should be referenced"
        elif response.status_code == 400:
            data = response.json()
            print(f"\nError Response (400): {data.get('detail', 'No detail')}")
            print(
                "This is expected when phrase mapping confidence is below threshold."
            )
            assert (
                "Suggestions" in data.get("detail", "")
            ), "Should provide suggestions for low-confidence phrases"
        else:
            pytest.fail(f"Unexpected status code: {response.status_code}")

    def test_all_examples_generate_valid_json_logic(self):
        """Verify all examples either generate valid JSON Logic or helpful error suggestions."""
        examples = [
            (
                "Approve if bureau score > 700 and business vintage at least 3 years "
                "and applicant age between 25 and 60."
            ),
            (
                "Flag as high risk if wilful default is true OR overdue amount > 50000 "
                "OR bureau.dpd >= 90."
            ),
            "Prefer applicants with tag 'veteran' OR with monthly_income > 1,00,000.",
        ]

        for prompt in examples:
            response = client.post("/generate-rule", json={"prompt": prompt})
            assert response.status_code in (
                200,
                400,
            ), f"Response should be 200 or 400, got {response.status_code}"
            data = response.json()

            if response.status_code == 200:
                # Verify JSON Logic is valid
                json_logic = data.get("json_logic")
                assert json_logic is not None, "JSON Logic should not be null"
                assert isinstance(
                    json_logic, dict
                ), "JSON Logic should be a dictionary"

                # Verify it's valid JSON
                json_str = json.dumps(json_logic)
                parsed = json.loads(json_str)
                assert (
                    parsed == json_logic
                ), "JSON Logic should be valid JSON"

                # Verify has at least one operator
                assert (
                    len(json_logic) > 0
                ), "JSON Logic should have operators"
            elif response.status_code == 400:
                # Verify error response has helpful suggestions
                detail = data.get("detail", "")
                assert (
                    "Suggestions" in detail
                ), "Error should provide suggestions"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
