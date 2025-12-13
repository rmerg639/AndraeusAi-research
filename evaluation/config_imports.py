"""
Centralized imports from andraeus/config.py for evaluation scripts.

This module provides a clean way for evaluation scripts to import
configuration defaults without duplicating values.

Usage in evaluation scripts:
    from config_imports import (
        BASE_MODEL,
        DEFAULT_LORA_CONFIG,
        DEFAULT_TRAINING_CONFIG,
        get_lora_config,
        get_bnb_config,
    )

Copyright (c) 2025 Rocco Andraeus Sergi. All Rights Reserved.
"""

import sys
from pathlib import Path

# Add parent directory to path so we can import from andraeus
_parent_dir = Path(__file__).parent.parent
if str(_parent_dir) not in sys.path:
    sys.path.insert(0, str(_parent_dir))

# Import from centralized config
from andraeus.config import (
    DEFAULT_MODEL_CONFIG,
    DEFAULT_LORA_CONFIG,
    DEFAULT_TRAINING_CONFIG,
    DEFAULT_EVAL_CONFIG,
    ModelConfig,
    LoRAConfig,
    TrainingConfig,
    EvalConfig,
    get_model_name,
)

# Convenience exports matching old naming conventions
BASE_MODEL = DEFAULT_MODEL_CONFIG.name

# Helper functions to create config objects for peft/transformers


def get_lora_config():
    """
    Get a LoraConfig object for peft using centralized defaults.

    Returns:
        peft.LoraConfig configured with default values
    """
    from peft import LoraConfig
    cfg = DEFAULT_LORA_CONFIG
    return LoraConfig(
        r=cfg.r,
        lora_alpha=cfg.lora_alpha,
        target_modules=cfg.target_modules,
        lora_dropout=cfg.lora_dropout,
        bias=cfg.bias,
        task_type=cfg.task_type,
    )


def get_bnb_config():
    """
    Get a BitsAndBytesConfig for 4-bit quantization using centralized defaults.

    Returns:
        transformers.BitsAndBytesConfig configured with default values
    """
    import torch
    from transformers import BitsAndBytesConfig
    cfg = DEFAULT_MODEL_CONFIG
    compute_dtype = getattr(torch, cfg.bnb_4bit_compute_dtype)
    return BitsAndBytesConfig(
        load_in_4bit=cfg.load_in_4bit,
        bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=cfg.bnb_4bit_use_double_quant,
    )


def get_training_args(output_dir: str, **overrides):
    """
    Get SFTConfig/TrainingArguments using centralized defaults.

    Args:
        output_dir: Directory for checkpoints and outputs
        **overrides: Any parameters to override defaults

    Returns:
        dict of training arguments (use with SFTConfig or TrainingArguments)
    """
    cfg = DEFAULT_TRAINING_CONFIG
    args = {
        "output_dir": output_dir,
        "num_train_epochs": cfg.num_train_epochs,
        "per_device_train_batch_size": cfg.per_device_train_batch_size,
        "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
        "learning_rate": cfg.learning_rate,
        "warmup_ratio": cfg.warmup_ratio,
        "bf16": cfg.bf16,
        "gradient_checkpointing": cfg.gradient_checkpointing,
        "optim": cfg.optim,
        "logging_steps": cfg.logging_steps,
        "save_strategy": cfg.save_strategy,
        "max_seq_length": cfg.max_length,
    }
    args.update(overrides)
    return args


# Export all for convenience
__all__ = [
    # Constants
    "BASE_MODEL",
    # Config objects
    "DEFAULT_MODEL_CONFIG",
    "DEFAULT_LORA_CONFIG",
    "DEFAULT_TRAINING_CONFIG",
    "DEFAULT_EVAL_CONFIG",
    # Config classes
    "ModelConfig",
    "LoRAConfig",
    "TrainingConfig",
    "EvalConfig",
    # Helper functions
    "get_model_name",
    "get_lora_config",
    "get_bnb_config",
    "get_training_args",
]
