"""
Andraeus AI Scaling and Context Window Solution Research

Zero-Context Personal Memory for Large Language Models.

Copyright (c) 2025 Rocco Andraeus Sergi. All Rights Reserved.
https://github.com/rmerg639/AndraeusAi-research
"""

__version__ = "2.2.0"
__author__ = "Rocco Andraeus Sergi"
__email__ = "andraeusbeats@gmail.com"
__license__ = "Proprietary"

from .core import (
    AndraeusConfig,
    AndraeusTrainer,
    load_model,
    train_personal_ai,
)

__all__ = [
    "__version__",
    "__author__",
    "AndraeusConfig",
    "AndraeusTrainer",
    "load_model",
    "train_personal_ai",
]
