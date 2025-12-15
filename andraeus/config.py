#!/usr/bin/env python3
"""
Centralized Configuration for Andraeus AI

All hyperparameters and settings in one place.
Edit this file instead of hardcoding values throughout the codebase.

Copyright (c) 2025 Rocco Andraeus Sergi. All Rights Reserved.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import os

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

@dataclass
class ModelConfig:
    """Base model configuration."""
    name: str = "Qwen/Qwen2.5-7B-Instruct"
    trust_remote_code: bool = True
    device_map: str = "auto"

    # Quantization (4-bit QLoRA)
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_use_double_quant: bool = True


# =============================================================================
# LORA CONFIGURATION
# =============================================================================

@dataclass
class LoRAConfig:
    """LoRA adapter configuration."""
    r: int = 64  # Rank - higher for better fact retention
    lora_alpha: int = 128  # 2x rank for stability
    lora_dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"

    # Target modules for Qwen2.5
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])


# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================

@dataclass
class TrainingConfig:
    """Training hyperparameters.

    NOTE: These values match core.py AndraeusConfig for consistency.
    If you change these, also update core.py.
    """
    # Core settings - MUST MATCH core.py AndraeusConfig
    num_train_epochs: int = 5
    per_device_train_batch_size: int = 2  # Matches core.py batch_size
    gradient_accumulation_steps: int = 4  # Matches core.py
    learning_rate: float = 2e-4  # Matches core.py
    warmup_ratio: float = 0.03  # Matches core.py

    # Memory optimization
    bf16: bool = True
    gradient_checkpointing: bool = True
    optim: str = "paged_adamw_8bit"

    # Logging and saving
    logging_steps: int = 10
    save_strategy: str = "epoch"

    # Data processing
    max_length: int = 512

    # Output
    output_dir: str = "./output/personal-ai"


# =============================================================================
# QUESTION VARIATION CONFIGURATION
# =============================================================================

@dataclass
class VariationConfig:
    """Question variation methodology settings."""
    variations_per_fact: int = 10  # Optimal based on ablation study
    include_typos: bool = True
    include_casual: bool = True
    include_formal: bool = True
    include_minimal: bool = True


# =============================================================================
# EVALUATION CONFIGURATION
# =============================================================================

@dataclass
class EvalConfig:
    """Evaluation settings."""
    # Sample sizes
    min_sample_size: int = 30  # testing standard
    bootstrap_iterations: int = 10000

    # Random seed for reproducibility
    random_seed: int = 42

    # Accuracy checking
    strict_accuracy: bool = True  # Use strict_accuracy_check

    # Generation settings
    max_new_tokens: int = 50
    temperature: float = 0.1
    do_sample: bool = False


# =============================================================================
# DEFAULT INSTANCES
# =============================================================================

# Use these throughout the codebase instead of hardcoding
DEFAULT_MODEL_CONFIG = ModelConfig()
DEFAULT_LORA_CONFIG = LoRAConfig()
DEFAULT_TRAINING_CONFIG = TrainingConfig()
DEFAULT_VARIATION_CONFIG = VariationConfig()
DEFAULT_EVAL_CONFIG = EvalConfig()


# =============================================================================
# ENVIRONMENT OVERRIDES
# =============================================================================

def get_model_name() -> str:
    """Get model name, allowing environment override."""
    return os.environ.get("ANDRAEUS_MODEL", DEFAULT_MODEL_CONFIG.name)


def get_output_dir() -> str:
    """Get output directory, allowing environment override."""
    return os.environ.get("ANDRAEUS_OUTPUT_DIR", DEFAULT_TRAINING_CONFIG.output_dir)


def get_variations_per_fact() -> int:
    """Get variations count, allowing environment override."""
    return int(os.environ.get("ANDRAEUS_VARIATIONS", DEFAULT_VARIATION_CONFIG.variations_per_fact))


# =============================================================================
# CONFIGURATION SUMMARY
# =============================================================================

def print_config():
    """Print current configuration for debugging."""
    print("=" * 60)
    print("ANDRAEUS CONFIGURATION")
    print("=" * 60)
    print(f"\nModel: {get_model_name()}")
    print(f"Output: {get_output_dir()}")
    print(f"Variations per fact: {get_variations_per_fact()}")
    print(f"\nLoRA: r={DEFAULT_LORA_CONFIG.r}, alpha={DEFAULT_LORA_CONFIG.lora_alpha}")
    print(f"Training: epochs={DEFAULT_TRAINING_CONFIG.num_train_epochs}, lr={DEFAULT_TRAINING_CONFIG.learning_rate}")
    print(f"Eval seed: {DEFAULT_EVAL_CONFIG.random_seed}")
    print("=" * 60)


if __name__ == "__main__":
    print_config()
