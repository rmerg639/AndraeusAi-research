"""
Andraeus AI - Personal Memory Fine-Tuning

A practical implementation of QLoRA for encoding personal facts into LLM weights.

NOTE: This is a standard technique implementation, not novel research.
See PAPER.md for limitations and honest assessment.

Copyright (c) 2025 Rocco Andraeus Sergi. All Rights Reserved.
https://github.com/rmerg639/AndraeusAi-research
"""

__version__ = "3.0.0"
__author__ = "Rocco Andraeus Sergi"
__email__ = "andraeusbeats@gmail.com"
__license__ = "Proprietary"

# Core functionality
from .core import (
    # Config and Trainer
    AndraeusConfig,
    AndraeusTrainer,
    # Model loading
    load_model,
    load_base_model,
    # Training functions
    train_personal_ai,
    quick_train,
    # Data generation
    generate_question_variations,
    prepare_training_data,
    # Evaluation
    strict_accuracy_check,
    # Exceptions
    AndraeusError,
    ValidationError,
    ModelLoadError,
    TrainingError,
)

# Centralized configuration
from .config import (
    ModelConfig,
    LoRAConfig,
    TrainingConfig,
    VariationConfig,
    EvalConfig,
    DEFAULT_MODEL_CONFIG,
    DEFAULT_LORA_CONFIG,
    DEFAULT_TRAINING_CONFIG,
    DEFAULT_VARIATION_CONFIG,
    DEFAULT_EVAL_CONFIG,
    get_model_name,
    get_output_dir,
    get_variations_per_fact,
)

__all__ = [
    # Metadata
    "__version__",
    "__author__",
    # Core classes
    "AndraeusConfig",
    "AndraeusTrainer",
    # Functions
    "load_model",
    "load_base_model",
    "train_personal_ai",
    "quick_train",
    "generate_question_variations",
    "prepare_training_data",
    "strict_accuracy_check",
    # Exceptions
    "AndraeusError",
    "ValidationError",
    "ModelLoadError",
    "TrainingError",
    # Config classes
    "ModelConfig",
    "LoRAConfig",
    "TrainingConfig",
    "VariationConfig",
    "EvalConfig",
    # Config defaults
    "DEFAULT_MODEL_CONFIG",
    "DEFAULT_LORA_CONFIG",
    "DEFAULT_TRAINING_CONFIG",
    "DEFAULT_VARIATION_CONFIG",
    "DEFAULT_EVAL_CONFIG",
    # Config helpers
    "get_model_name",
    "get_output_dir",
    "get_variations_per_fact",
]
