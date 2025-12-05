"""Tests for the mapper module."""

import pytest

from src.core.mapper import KeyMapper, PhraseExtractor


class TestPhraseExtractor:
    """Tests for PhraseExtractor."""

    def test_extract_phrases_with_quotes(self):
        """Test extraction of quoted phrases."""
        text = 'The user said "premium member"'
        phrases = PhraseExtractor.extract_phrases(text)
        assert "premium member" in phrases

    def test_extract_phrases_with_capitalized_terms(self):
        """Test extraction of capitalized terms."""
        text = "Check if PremiumMember status is Active"
        phrases = PhraseExtractor.extract_phrases(text)
        assert any("Premium" in p or "Active" in p for p in phrases)

    def test_extract_numeric_values(self):
        """Test extraction of numeric values."""
        text = "Greater than 100 dollars"
        values = PhraseExtractor.extract_numeric_values(text)
        assert len(values) > 0
        assert any(v[0] == 100.0 for v in values)

    def test_extract_numeric_ranges(self):
        """Test extraction of numeric ranges."""
        text = "Between 50 and 100 purchases"
        values = PhraseExtractor.extract_numeric_values(text)
        assert len(values) > 0

    def test_extract_booleans_positive(self):
        """Test extraction of positive boolean indicators."""
        text = "must be active and verified"
        booleans = PhraseExtractor.extract_booleans(text)
        assert len(booleans) > 0

    def test_extract_booleans_negative(self):
        """Test extraction of negative boolean indicators."""
        text = "cannot be suspended or banned"
        booleans = PhraseExtractor.extract_booleans(text)
        assert any(not b[1] for b in booleans)


class TestKeyMapper:
    """Tests for KeyMapper."""

    @pytest.fixture
    def mapper(self):
        """Create a mapper for testing."""
        from src.core.embeddings import EmbeddingConfig, EmbeddingManager
        from src.core.vector_store import InMemoryVectorStore

        config = EmbeddingConfig(provider="sentence-transformers")
        embedding_manager = EmbeddingManager(config)
        vector_store = InMemoryVectorStore()

        store_keys = ["user_age", "purchase_amount", "is_premium_member", "user_status"]
        mapper = KeyMapper(
            embedding_manager,
            vector_store,
            store_keys,
            similarity_threshold=0.5
        )
        return mapper

    def test_mapper_initialization(self, mapper):
        """Test mapper initialization."""
        assert mapper is not None
        assert len(mapper.store_keys) > 0

    def test_map_phrases(self, mapper):
        """Test phrase mapping."""
        text = "Check if the user age is greater than 18"
        mappings, errors = mapper.map_phrases(text)
        assert len(mappings) > 0 or len(errors) > 0

    def test_map_phrases_with_high_threshold(self, mapper):
        """Test phrase mapping with high threshold."""
        mapper.similarity_threshold = 0.99
        text = "Check premium member status"
        mappings, errors = mapper.map_phrases(text)

    def test_get_potential_matches(self, mapper):
        """Test getting potential matches."""
        phrase = "age"
        matches = mapper.get_all_potential_matches(phrase, top_k=2)
        assert len(matches) > 0
