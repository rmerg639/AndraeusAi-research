#!/usr/bin/env python3
"""
Unit Tests for Configuration Modules

Tests for:
- andraeus/config.py centralized configuration
- evaluation/config_imports.py helper module
- Configuration consistency across the codebase

Run with: pytest tests/test_config.py -v

Copyright (c) 2025 Rocco Andraeus Sergi. All Rights Reserved.
"""

import pytest
import sys
from pathlib import Path

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "evaluation"))


# =============================================================================
# ANDRAEUS CONFIG TESTS
# =============================================================================

class TestAndraeusConfig:
    """Tests for andraeus/config.py."""

    def test_default_model_config_exists(self):
        """Test that DEFAULT_MODEL_CONFIG exists and has required fields."""
        from andraeus.config import DEFAULT_MODEL_CONFIG
        assert hasattr(DEFAULT_MODEL_CONFIG, 'name')
        assert hasattr(DEFAULT_MODEL_CONFIG, 'load_in_4bit')

    def test_default_model_name(self):
        """Test default model name."""
        from andraeus.config import DEFAULT_MODEL_CONFIG
        assert DEFAULT_MODEL_CONFIG.name == "Qwen/Qwen2.5-7B-Instruct"

    def test_default_lora_config_exists(self):
        """Test that DEFAULT_LORA_CONFIG exists and has required fields."""
        from andraeus.config import DEFAULT_LORA_CONFIG
        assert hasattr(DEFAULT_LORA_CONFIG, 'r')
        assert hasattr(DEFAULT_LORA_CONFIG, 'lora_alpha')
        assert hasattr(DEFAULT_LORA_CONFIG, 'target_modules')
        assert hasattr(DEFAULT_LORA_CONFIG, 'lora_dropout')

    def test_lora_config_values(self):
        """Test LoRA configuration values."""
        from andraeus.config import DEFAULT_LORA_CONFIG
        assert DEFAULT_LORA_CONFIG.r == 64
        assert DEFAULT_LORA_CONFIG.lora_alpha == 128
        assert DEFAULT_LORA_CONFIG.lora_dropout == 0.05

    def test_lora_target_modules(self):
        """Test LoRA target modules include required projections."""
        from andraeus.config import DEFAULT_LORA_CONFIG
        required_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
        for module in required_modules:
            assert module in DEFAULT_LORA_CONFIG.target_modules

    def test_training_config_exists(self):
        """Test that DEFAULT_TRAINING_CONFIG exists."""
        from andraeus.config import DEFAULT_TRAINING_CONFIG
        assert hasattr(DEFAULT_TRAINING_CONFIG, 'num_train_epochs')
        assert hasattr(DEFAULT_TRAINING_CONFIG, 'learning_rate')

    def test_training_config_values(self):
        """Test training configuration values match core.py."""
        from andraeus.config import DEFAULT_TRAINING_CONFIG
        assert DEFAULT_TRAINING_CONFIG.num_train_epochs == 5
        assert DEFAULT_TRAINING_CONFIG.per_device_train_batch_size == 2
        assert DEFAULT_TRAINING_CONFIG.gradient_accumulation_steps == 4
        assert DEFAULT_TRAINING_CONFIG.learning_rate == 2e-4
        assert DEFAULT_TRAINING_CONFIG.warmup_ratio == 0.03

    def test_get_model_name_function(self):
        """Test get_model_name function."""
        from andraeus.config import get_model_name
        assert get_model_name() == "Qwen/Qwen2.5-7B-Instruct"

    def test_get_variations_per_fact(self):
        """Test get_variations_per_fact function."""
        from andraeus.config import get_variations_per_fact
        assert get_variations_per_fact() == 10


# =============================================================================
# CONFIG IMPORTS TESTS
# =============================================================================

class TestConfigImports:
    """Tests for evaluation/config_imports.py."""

    def test_base_model_import(self):
        """Test BASE_MODEL can be imported."""
        from config_imports import BASE_MODEL
        assert BASE_MODEL == "Qwen/Qwen2.5-7B-Instruct"

    def test_get_lora_config_function(self):
        """Test get_lora_config returns a LoraConfig object."""
        from config_imports import get_lora_config
        config = get_lora_config()
        # Check it has the expected attributes
        assert hasattr(config, 'r')
        assert hasattr(config, 'lora_alpha')
        assert config.r == 64
        assert config.lora_alpha == 128

    def test_get_lora_config_type(self):
        """Test get_lora_config returns correct type."""
        from config_imports import get_lora_config
        from peft import LoraConfig
        config = get_lora_config()
        assert isinstance(config, LoraConfig)

    def test_get_bnb_config_function(self):
        """Test get_bnb_config returns a BitsAndBytesConfig object."""
        from config_imports import get_bnb_config
        config = get_bnb_config()
        assert hasattr(config, 'load_in_4bit')
        assert config.load_in_4bit == True

    def test_config_consistency(self):
        """Test that config_imports matches andraeus/config.py."""
        from config_imports import BASE_MODEL, get_lora_config
        from andraeus.config import DEFAULT_MODEL_CONFIG, DEFAULT_LORA_CONFIG

        assert BASE_MODEL == DEFAULT_MODEL_CONFIG.name

        lora = get_lora_config()
        assert lora.r == DEFAULT_LORA_CONFIG.r
        assert lora.lora_alpha == DEFAULT_LORA_CONFIG.lora_alpha
        assert lora.lora_dropout == DEFAULT_LORA_CONFIG.lora_dropout


# =============================================================================
# CORE MODULE CONFIG TESTS
# =============================================================================

class TestCoreConfig:
    """Tests for andraeus/core.py AndraeusConfig."""

    def test_andraeus_config_dataclass(self):
        """Test AndraeusConfig exists and is configurable."""
        from andraeus.core import AndraeusConfig
        config = AndraeusConfig()
        assert hasattr(config, 'base_model')
        assert hasattr(config, 'lora_r')
        assert hasattr(config, 'lora_alpha')

    def test_default_values(self):
        """Test AndraeusConfig default values."""
        from andraeus.core import AndraeusConfig
        config = AndraeusConfig()
        assert config.base_model == "Qwen/Qwen2.5-7B-Instruct"
        assert config.lora_r == 64
        assert config.lora_alpha == 128
        assert config.variations_per_fact == 10
        assert config.num_epochs == 5

    def test_custom_values(self):
        """Test AndraeusConfig with custom values."""
        from andraeus.core import AndraeusConfig
        config = AndraeusConfig(
            lora_r=32,
            lora_alpha=64,
            variations_per_fact=5
        )
        assert config.lora_r == 32
        assert config.lora_alpha == 64
        assert config.variations_per_fact == 5

    def test_config_matches_centralized(self):
        """Test that AndraeusConfig defaults match centralized config."""
        from andraeus.core import AndraeusConfig
        from andraeus.config import DEFAULT_MODEL_CONFIG, DEFAULT_LORA_CONFIG

        config = AndraeusConfig()
        assert config.base_model == DEFAULT_MODEL_CONFIG.name
        assert config.lora_r == DEFAULT_LORA_CONFIG.r
        assert config.lora_alpha == DEFAULT_LORA_CONFIG.lora_alpha


# =============================================================================
# CROSS-MODULE CONSISTENCY TESTS
# =============================================================================

class TestConfigConsistency:
    """Tests for configuration consistency across the codebase."""

    def test_model_name_consistent(self):
        """Test model name is consistent across all config sources."""
        from andraeus.config import DEFAULT_MODEL_CONFIG, get_model_name
        from andraeus.core import AndraeusConfig
        from config_imports import BASE_MODEL

        expected = "Qwen/Qwen2.5-7B-Instruct"
        assert DEFAULT_MODEL_CONFIG.name == expected
        assert get_model_name() == expected
        assert AndraeusConfig().base_model == expected
        assert BASE_MODEL == expected

    def test_lora_r_consistent(self):
        """Test LoRA r value is consistent."""
        from andraeus.config import DEFAULT_LORA_CONFIG
        from andraeus.core import AndraeusConfig
        from config_imports import get_lora_config

        expected_r = 64
        assert DEFAULT_LORA_CONFIG.r == expected_r
        assert AndraeusConfig().lora_r == expected_r
        assert get_lora_config().r == expected_r

    def test_lora_alpha_consistent(self):
        """Test LoRA alpha value is consistent."""
        from andraeus.config import DEFAULT_LORA_CONFIG
        from andraeus.core import AndraeusConfig
        from config_imports import get_lora_config

        expected_alpha = 128
        assert DEFAULT_LORA_CONFIG.lora_alpha == expected_alpha
        assert AndraeusConfig().lora_alpha == expected_alpha
        assert get_lora_config().lora_alpha == expected_alpha

    def test_variations_consistent(self):
        """Test variations_per_fact is consistent."""
        from andraeus.config import get_variations_per_fact
        from andraeus.core import AndraeusConfig

        expected = 10
        assert get_variations_per_fact() == expected
        assert AndraeusConfig().variations_per_fact == expected


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
