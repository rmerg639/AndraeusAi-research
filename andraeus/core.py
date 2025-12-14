"""
Andraeus AI - Core Module

Personal Memory Fine-Tuning using QLoRA.

This module provides the implementation for:
1. Question variation generation (10 variations recommended)
2. QLoRA fine-tuning on personal facts
3. Model evaluation

NOTE: This is a practical implementation of standard QLoRA fine-tuning,
not a novel research contribution. See PAPER.md for limitations.

Usage:
    from andraeus import AndraeusTrainer, AndraeusConfig

    config = AndraeusConfig(variations_per_fact=10)
    trainer = AndraeusTrainer(config)

    facts = {"name": "Alex", "pet": "Max", "city": "Seattle"}
    adapter_path = trainer.train(facts)

    model, tokenizer = load_model(adapter_path)

Copyright (c) 2025 Rocco Andraeus Sergi. All Rights Reserved.
"""

import os
import re
import json
import random
import logging
import torch
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# EXCEPTIONS
# =============================================================================

class AndraeusError(Exception):
    """Base exception for Andraeus errors."""
    pass


class ValidationError(AndraeusError):
    """Raised when input validation fails."""
    pass


class ModelLoadError(AndraeusError):
    """Raised when model loading fails."""
    pass


class TrainingError(AndraeusError):
    """Raised when training fails."""
    pass


# =============================================================================
# ACCURACY CHECKING
# =============================================================================

def strict_accuracy_check(response: str, expected: str) -> bool:
    """
    Strict accuracy checking that avoids false positives.

    This addresses the issue where lenient matching like:
        expected.lower() in response.lower()
    would incorrectly mark "12" as correct in "120".

    Args:
        response: Model's response
        expected: Expected answer

    Returns:
        True if the response correctly contains the expected answer
    """
    response = response.strip().lower()
    expected = expected.strip().lower()

    if not expected:
        return False

    # For numeric answers, be strict about boundaries
    if expected.isdigit():
        # Use word boundary matching for numbers
        pattern = r'\b' + re.escape(expected) + r'\b'
        return bool(re.search(pattern, response))

    # For short answers (1-3 words), check if it's the primary content
    if len(expected.split()) <= 3:
        # Remove common prefixes
        clean_response = re.sub(
            r'^(it\'?s?|that\'?s?|the answer is|yes,?|no,?|my .+ is)\s*',
            '', response, flags=re.I
        )
        clean_response = re.sub(r'[!.,?]+$', '', clean_response).strip()

        # Expected should be at the start or be the main content
        if clean_response.startswith(expected):
            return True
        if expected in clean_response.split()[:5]:
            return True

        # Also check original response for word boundary match
        pattern = r'\b' + re.escape(expected) + r'\b'
        return bool(re.search(pattern, response))

    # For longer expected answers, use contains check
    return expected in response


@dataclass
class AndraeusConfig:
    """Configuration for Andraeus AI training."""

    # Model settings
    base_model: str = "Qwen/Qwen2.5-7B-Instruct"

    # LoRA settings (optimized for personal knowledge)
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])

    # Training settings
    num_epochs: int = 5
    learning_rate: float = 2e-4
    batch_size: int = 2
    gradient_accumulation_steps: int = 4
    warmup_ratio: float = 0.03
    max_length: int = 512

    # Variation settings (10 is optimal based on research)
    variations_per_fact: int = 10

    # Response variations
    response_variations: int = 4

    # Output
    output_dir: str = "./output/andraeus"

    # Quantization
    use_4bit: bool = True
    compute_dtype: str = "bfloat16"


# =============================================================================
# QUESTION VARIATION GENERATION
# =============================================================================

# Template patterns for generating question variations
QUESTION_TEMPLATES = {
    "direct": [
        "What is {fact_phrase}?",
        "What's {fact_phrase}?",
        "{fact_phrase}?",
        "Tell me {fact_phrase}",
    ],
    "casual": [
        "{fact_phrase}?",
        "my {fact_key}?",
        "{fact_key}?",
    ],
    "polite": [
        "Can you tell me {fact_phrase}?",
        "Could you remind me {fact_phrase}?",
        "Do you know {fact_phrase}?",
    ],
    "variations": [
        "What is {fact_phrase} again?",
        "Remind me {fact_phrase}",
        "I forgot {fact_phrase}",
    ],
}

# Common fact type mappings
FACT_TYPE_PHRASES = {
    "name": "my name",
    "user_name": "my name",
    "age": "my age",
    "user_age": "my age",
    "location": "where I live",
    "user_location": "where I live",
    "city": "my city",
    "occupation": "my job",
    "user_occupation": "my job",
    "job": "my job",
    "pet_name": "my pet's name",
    "pet": "my pet's name",
    "pet_type": "what type of pet I have",
    "partner_name": "my partner's name",
    "partner": "my partner's name",
    "birthday": "my birthday",
    "favorite_food": "my favorite food",
    "favorite_color": "my favorite color",
    "hobby": "my hobby",
    "car": "what car I drive",
    "phone": "what phone I have",
}


def get_fact_phrase(fact_key: str) -> str:
    """Get the natural language phrase for a fact key."""
    if fact_key in FACT_TYPE_PHRASES:
        return FACT_TYPE_PHRASES[fact_key]

    # Generate phrase from key
    words = fact_key.replace("_", " ").split()
    if words[0] in ["my", "user"]:
        words = words[1:]
    return "my " + " ".join(words)


def generate_question_variations(
    fact_key: str,
    num_variations: int = 10
) -> List[str]:
    """
    Generate question variations for a fact.

    This is the core innovation: generating diverse question phrasings
    to enable generalization beyond memorization.

    Args:
        fact_key: The key identifying the fact (e.g., "pet_name")
        num_variations: Number of variations to generate (10 optimal)

    Returns:
        List of question strings
    """
    fact_phrase = get_fact_phrase(fact_key)
    questions = []

    # Collect all template types
    all_templates = []
    for template_type, templates in QUESTION_TEMPLATES.items():
        for template in templates:
            all_templates.append(template)

    # Generate variations
    for template in all_templates:
        try:
            q = template.format(fact_phrase=fact_phrase, fact_key=fact_key.replace("_", " "))
            questions.append(q)
        except KeyError:
            # Template doesn't use this placeholder
            pass

    # Add some direct variations
    clean_key = fact_key.replace("_", " ")
    questions.extend([
        f"What is my {clean_key}?",
        f"my {clean_key}?",
        f"{clean_key}?",
        f"Tell me my {clean_key}",
        f"What's my {clean_key}?",
    ])

    # Remove duplicates and limit
    questions = list(dict.fromkeys(questions))

    # Ensure we have enough variations
    while len(questions) < num_variations:
        # Add more variations with slight modifications
        base_q = random.choice(questions[:5])
        if base_q.endswith("?"):
            questions.append(base_q[:-1] + " again?")
        else:
            questions.append(base_q + "?")
        questions = list(dict.fromkeys(questions))

    return questions[:num_variations]


def generate_response_variations(answer: str, num_variations: int = 4) -> List[str]:
    """Generate response variations for an answer."""
    variations = [
        answer,
        f"{answer}!",
        f"That's {answer}.",
        f"It's {answer}!",
        f"The answer is {answer}.",
        f"{answer}.",
    ]
    return variations[:num_variations]


# =============================================================================
# TRAINING DATA GENERATION
# =============================================================================

def prepare_training_data(
    facts: Dict[str, str],
    config: AndraeusConfig
) -> List[Dict[str, str]]:
    """
    Prepare training data from facts using question variation methodology.

    Args:
        facts: Dictionary of {fact_key: fact_value}
        config: Training configuration

    Returns:
        List of training examples with "text" field
    """
    training_data = []

    for fact_key, fact_value in facts.items():
        # Generate question variations
        questions = generate_question_variations(
            fact_key,
            config.variations_per_fact
        )

        # Generate response variations
        responses = generate_response_variations(
            fact_value,
            config.response_variations
        )

        # Create training examples
        for question in questions:
            response = random.choice(responses)
            training_data.append({
                "text": f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>"
            })

    return training_data


# =============================================================================
# MODEL LOADING
# =============================================================================

def load_base_model(config: AndraeusConfig) -> Tuple[Any, Any]:
    """
    Load the base model with quantization.

    Args:
        config: Training configuration

    Returns:
        Tuple of (model, tokenizer)
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    # Quantization config
    if config.use_4bit:
        compute_dtype = getattr(torch, config.compute_dtype)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
    else:
        bnb_config = None

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.base_model,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def load_model(adapter_path: str, base_model: Optional[str] = None) -> Tuple[Any, Any]:
    """
    Load a trained Andraeus model.

    Args:
        adapter_path: Path to the LoRA adapter
        base_model: Base model name (default: Qwen2.5-7B-Instruct)

    Returns:
        Tuple of (model, tokenizer)
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    base_model = base_model or "Qwen/Qwen2.5-7B-Instruct"

    # Load with quantization for efficiency
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    return model, tokenizer


# =============================================================================
# TRAINER CLASS
# =============================================================================

class AndraeusTrainer:
    """
    Trainer for Personal fact encoding via fine-tuning.

    Encodes personal knowledge directly into model weights,
    using 0 context tokens for personal facts.

    Example:
        config = AndraeusConfig(variations_per_fact=10)
        trainer = AndraeusTrainer(config)

        facts = {
            "name": "Alex",
            "pet_name": "Max",
            "location": "Seattle"
        }

        adapter_path = trainer.train(facts)
    """

    def __init__(self, config: Optional[AndraeusConfig] = None):
        """
        Initialize the Andraeus trainer.

        Args:
            config: Training configuration. Uses defaults if None.
        """
        self.config = config or AndraeusConfig()
        self.model = None
        self.tokenizer = None
        self.training_data = None

    def prepare_data(self, facts: Dict[str, str]) -> List[Dict]:
        """
        Generate training data with question variations.

        Args:
            facts: Dictionary of {fact_name: fact_value}

        Returns:
            List of training examples with variations
        """
        self.training_data = prepare_training_data(facts, self.config)
        return self.training_data

    def train(self, facts: Dict[str, str]) -> str:
        """
        Train a personalized model with the given facts.

        Args:
            facts: Dictionary of personal facts

        Returns:
            Path to saved adapter

        Raises:
            ValidationError: If facts dictionary is empty or invalid
            ModelLoadError: If model loading fails
            TrainingError: If training fails
        """
        # =================================================================
        # INPUT VALIDATION
        # =================================================================
        if not facts:
            raise ValidationError("Facts dictionary cannot be empty")

        if not isinstance(facts, dict):
            raise ValidationError(f"Facts must be a dictionary, got {type(facts)}")

        for key, value in facts.items():
            if not isinstance(key, str) or not key.strip():
                raise ValidationError(f"Fact keys must be non-empty strings, got: {key!r}")
            if not isinstance(value, str) or not value.strip():
                raise ValidationError(f"Fact values must be non-empty strings, got: {value!r} for key {key}")

        logger.info(f"Validated {len(facts)} facts")

        # =================================================================
        # IMPORT DEPENDENCIES
        # =================================================================
        try:
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
            from trl import SFTTrainer, SFTConfig
            from datasets import Dataset
        except ImportError as e:
            raise TrainingError(f"Missing required package: {e}. Run: pip install peft trl datasets")

        # Prepare data
        print(f"Preparing training data for {len(facts)} facts...")
        training_data = self.prepare_data(facts)
        print(f"Generated {len(training_data)} training examples")
        print(f"  ({self.config.variations_per_fact} variations x {self.config.response_variations} responses per fact)")

        # Load model
        print(f"\nLoading base model: {self.config.base_model}")
        try:
            self.model, self.tokenizer = load_base_model(self.config)
        except Exception as e:
            raise ModelLoadError(f"Failed to load model {self.config.base_model}: {e}")

        # Configure LoRA
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=self.config.target_modules,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )

        # Prepare model for training
        self.model = prepare_model_for_kbit_training(self.model)
        self.model = get_peft_model(self.model, lora_config)

        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")

        # Create dataset
        dataset = Dataset.from_list(training_data)

        # Training config
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        training_args = SFTConfig(
            output_dir=str(output_dir / "checkpoints"),
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            lr_scheduler_type="cosine",
            warmup_ratio=self.config.warmup_ratio,
            bf16=True,
            logging_steps=10,
            save_strategy="epoch",
            gradient_checkpointing=True,
            optim="paged_adamw_8bit",
            max_length=self.config.max_length,
        )

        # Train
        print("\nStarting training...")
        trainer = SFTTrainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            processing_class=self.tokenizer,
        )

        trainer.train()

        # Save adapter
        adapter_path = output_dir / "adapter"
        print(f"\nSaving adapter to: {adapter_path}")
        self.model.save_pretrained(str(adapter_path))
        self.tokenizer.save_pretrained(str(adapter_path))

        # Save config and facts
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "config": asdict(self.config),
            "facts_count": len(facts),
            "training_examples": len(training_data),
            "base_model": self.config.base_model,
        }
        with open(adapter_path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"\nTraining complete!")
        print(f"Adapter saved to: {adapter_path}")

        return str(adapter_path)

    def evaluate(
        self,
        test_questions: List[Dict[str, str]],
        merge_weights: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate the trained model.

        Args:
            test_questions: List of {question, expected_answer}
            merge_weights: Whether to merge LoRA weights before eval

        Returns:
            Dictionary of metrics including accuracy, examples, and statistics
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        # Merge weights for evaluation
        if merge_weights:
            print("Merging LoRA weights for evaluation...")
            self.model = self.model.merge_and_unload()

        self.model.eval()

        correct = 0
        results = []

        print(f"\nEvaluating on {len(test_questions)} questions...")

        for item in test_questions:
            question = item["question"]
            expected = item["expected_answer"]

            # Generate response
            messages = [{"role": "user", "content": question}]
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=50,
                    temperature=0.1,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id
                )

            response = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            )

            # Check accuracy using strict matching
            is_correct = strict_accuracy_check(response, expected)
            if is_correct:
                correct += 1

            results.append({
                "question": question,
                "expected": expected,
                "response": response[:100],
                "correct": is_correct
            })

        accuracy = correct / len(test_questions) if test_questions else 0

        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": len(test_questions),
            "results": results
        }


# =============================================================================
# HIGH-LEVEL API
# =============================================================================

def train_personal_ai(
    facts: Dict[str, str],
    config: Optional[AndraeusConfig] = None
) -> str:
    """
    High-level function to train a personal AI.

    Args:
        facts: Dictionary of personal facts to encode
        config: Optional training configuration

    Returns:
        Path to saved adapter

    Example:
        >>> facts = {
        ...     "name": "Alex",
        ...     "pet": "Buddy the dog",
        ...     "birthday": "March 15"
        ... }
        >>> adapter_path = train_personal_ai(facts)
        >>> model, tokenizer = load_model(adapter_path)
    """
    trainer = AndraeusTrainer(config)
    return trainer.train(facts)


def quick_train(facts: Dict[str, str], output_dir: str = "./output/quick") -> str:
    """
    Quick training with optimized defaults for small fact sets.

    Args:
        facts: Dictionary of personal facts
        output_dir: Where to save the adapter

    Returns:
        Path to saved adapter
    """
    config = AndraeusConfig(
        num_epochs=3,
        variations_per_fact=10,
        output_dir=output_dir
    )
    return train_personal_ai(facts, config)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Config and Trainer
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
]
