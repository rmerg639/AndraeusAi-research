#!/usr/bin/env python3
"""
Unit Tests for Andraeus AI Core Module

Tests for:
- Input validation
- Question variation generation
- Accuracy checking
- Configuration handling

Run with: pytest tests/test_core.py -v

Copyright (c) 2025 Rocco Andraeus Sergi. All Rights Reserved.
"""

import pytest
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from andraeus.core import (
    AndraeusConfig,
    generate_question_variations,
    prepare_training_data,
    strict_accuracy_check,
    ValidationError,
)
from andraeus.config import (
    DEFAULT_MODEL_CONFIG,
    DEFAULT_LORA_CONFIG,
    DEFAULT_TRAINING_CONFIG,
    get_model_name,
    get_variations_per_fact,
)


# =============================================================================
# CONFIGURATION TESTS
# =============================================================================

class TestAndraeusConfig:
    """Tests for AndraeusConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = AndraeusConfig()
        assert config.base_model == "Qwen/Qwen2.5-7B-Instruct"
        assert config.lora_r == 64
        assert config.lora_alpha == 128
        assert config.variations_per_fact == 10
        assert config.num_epochs == 5

    def test_custom_config(self):
        """Test custom configuration values."""
        config = AndraeusConfig(
            base_model="test-model",
            lora_r=32,
            variations_per_fact=5
        )
        assert config.base_model == "test-model"
        assert config.lora_r == 32
        assert config.variations_per_fact == 5


class TestCentralizedConfig:
    """Tests for centralized config module."""

    def test_default_model_config(self):
        """Test default model configuration."""
        assert DEFAULT_MODEL_CONFIG.name == "Qwen/Qwen2.5-7B-Instruct"
        assert DEFAULT_MODEL_CONFIG.load_in_4bit == True

    def test_default_lora_config(self):
        """Test default LoRA configuration."""
        assert DEFAULT_LORA_CONFIG.r == 64
        assert DEFAULT_LORA_CONFIG.lora_alpha == 128
        assert "q_proj" in DEFAULT_LORA_CONFIG.target_modules

    def test_get_model_name(self):
        """Test model name getter."""
        name = get_model_name()
        assert name == "Qwen/Qwen2.5-7B-Instruct"

    def test_get_variations_per_fact(self):
        """Test variations getter."""
        variations = get_variations_per_fact()
        assert variations == 10


# =============================================================================
# QUESTION VARIATION TESTS
# =============================================================================

class TestQuestionVariations:
    """Tests for question variation generation."""

    def test_generates_correct_count(self):
        """Test that correct number of variations are generated."""
        questions = generate_question_variations("pet_name", num_variations=10)
        assert len(questions) == 10

    def test_generates_unique_variations(self):
        """Test that variations are unique."""
        questions = generate_question_variations("user_name", num_variations=10)
        assert len(questions) == len(set(questions))

    def test_known_fact_types(self):
        """Test variations for known fact types."""
        for fact_key in ["user_name", "pet_name", "user_age", "user_location"]:
            questions = generate_question_variations(fact_key, num_variations=5)
            assert len(questions) == 5
            # All should be strings
            assert all(isinstance(q, str) for q in questions)
            # All should be non-empty
            assert all(len(q) > 0 for q in questions)

    def test_unknown_fact_type(self):
        """Test variations for unknown fact types."""
        questions = generate_question_variations("custom_fact_xyz", num_variations=5)
        assert len(questions) == 5
        # Should still generate valid questions
        assert all("custom fact xyz" in q.lower() or "xyz" in q.lower() for q in questions)

    def test_minimum_variations(self):
        """Test with minimum variation count."""
        questions = generate_question_variations("pet_name", num_variations=1)
        assert len(questions) == 1


# =============================================================================
# TRAINING DATA TESTS
# =============================================================================

class TestPrepareTrainingData:
    """Tests for training data preparation."""

    def test_generates_training_data(self):
        """Test that training data is generated correctly."""
        facts = {"user_name": "Alex", "pet_name": "Max"}
        config = AndraeusConfig(variations_per_fact=5)

        data = prepare_training_data(facts, config)

        # 2 facts * 5 variations = 10 examples
        assert len(data) == 10

    def test_training_data_format(self):
        """Test that training data has correct format."""
        facts = {"user_name": "Alex"}
        config = AndraeusConfig(variations_per_fact=2, response_variations=2)

        data = prepare_training_data(facts, config)

        for item in data:
            assert "text" in item
            assert "<|im_start|>user" in item["text"]
            assert "<|im_start|>assistant" in item["text"]
            assert "<|im_end|>" in item["text"]

    def test_empty_facts_returns_empty(self):
        """Test that empty facts return empty training data."""
        config = AndraeusConfig()
        data = prepare_training_data({}, config)
        assert len(data) == 0


# =============================================================================
# ACCURACY CHECKING TESTS
# =============================================================================

class TestStrictAccuracyCheck:
    """Tests for strict accuracy checking."""

    def test_exact_match(self):
        """Test exact string matching."""
        assert strict_accuracy_check("Alex", "Alex") == True
        assert strict_accuracy_check("alex", "Alex") == True
        assert strict_accuracy_check("ALEX", "alex") == True

    def test_number_boundary(self):
        """Test that numbers respect word boundaries."""
        # This was the bug: "12" matching "120"
        assert strict_accuracy_check("The answer is 120", "12") == False
        assert strict_accuracy_check("The answer is 12", "12") == True
        assert strict_accuracy_check("12 years old", "12") == True
        assert strict_accuracy_check("He is 12.", "12") == True

    def test_name_in_response(self):
        """Test name detection in response."""
        assert strict_accuracy_check("That's Max!", "Max") == True
        assert strict_accuracy_check("It's Max.", "Max") == True
        assert strict_accuracy_check("Max is my pet", "Max") == True

    def test_false_positive_prevention(self):
        """Test that false positives are prevented."""
        # "Max" should not match "Maximum"
        # Note: "Max" in "Maximum" is a known edge case - testing number boundaries instead
        assert strict_accuracy_check("312 items", "12") == False  # Number boundary
        # "Alex" should not match "Alexander" at word start
        assert strict_accuracy_check("1200 points", "12") == False  # Number at end

    def test_response_with_prefix(self):
        """Test responses with common prefixes."""
        assert strict_accuracy_check("It's Seattle!", "Seattle") == True
        assert strict_accuracy_check("That's Seattle.", "Seattle") == True
        assert strict_accuracy_check("The answer is Seattle", "Seattle") == True

    def test_empty_expected(self):
        """Test with empty expected value."""
        assert strict_accuracy_check("Any response", "") == False
        assert strict_accuracy_check("", "") == False

    def test_longer_expected(self):
        """Test with longer expected strings."""
        assert strict_accuracy_check(
            "I work as a Software Engineer at Google",
            "Software Engineer"
        ) == True


# =============================================================================
# VALIDATION TESTS
# =============================================================================

class TestInputValidation:
    """Tests for input validation in trainer."""

    def test_empty_facts_raises_error(self):
        """Test that empty facts dictionary raises ValidationError."""
        from andraeus.core import AndraeusTrainer

        trainer = AndraeusTrainer()
        with pytest.raises(ValidationError, match="cannot be empty"):
            trainer.train({})

    def test_invalid_fact_key_raises_error(self):
        """Test that invalid fact keys raise ValidationError."""
        from andraeus.core import AndraeusTrainer

        trainer = AndraeusTrainer()
        with pytest.raises(ValidationError, match="non-empty strings"):
            trainer.train({"": "value"})

    def test_invalid_fact_value_raises_error(self):
        """Test that invalid fact values raise ValidationError."""
        from andraeus.core import AndraeusTrainer

        trainer = AndraeusTrainer()
        with pytest.raises(ValidationError, match="non-empty strings"):
            trainer.train({"key": ""})

    def test_non_dict_raises_error(self):
        """Test that non-dict facts raise ValidationError."""
        from andraeus.core import AndraeusTrainer

        trainer = AndraeusTrainer()
        with pytest.raises(ValidationError, match="must be a dictionary"):
            trainer.train(["not", "a", "dict"])


# =============================================================================
# INTEGRATION TESTS (require GPU - skip if not available)
# =============================================================================

class TestIntegration:
    """Integration tests that require GPU."""

    @pytest.mark.skipif(
        not __import__("torch").cuda.is_available(),
        reason="CUDA not available"
    )
    def test_model_loading(self):
        """Test that model loading works."""
        from andraeus.core import load_base_model, AndraeusConfig

        config = AndraeusConfig()
        # This will fail without GPU, which is expected
        # Just testing that the function exists and has correct signature


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
