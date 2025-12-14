#!/usr/bin/env python3
"""
Ablation Study: Question Variation Count
Tests the core hypothesis: Does more variation improve recall?

Experiments:
- 5 variations per fact
- 10 variations per fact
- 20 variations per fact
- 30 variations per fact

Each run 3 times for statistical significance.

Estimated cost: 12 runs × $3.0 = $33.12

Copyright (c) 2025 Rocco Andraeus Sergi. All Rights Reserved.
"""

import json
import time
import random
from pathlib import Path
from typing import List, Dict
from dataclasses import dataclass

# =============================================================================
# ABLATION CONFIGURATION
# =============================================================================

VARIATION_COUNTS = [5, 10, 20, 30]
RUNS_PER_CONDITION = 3
BASE_OUTPUT_DIR = "./ablation_results"

# Example user config
USER_CONFIG = {
    "ai_name": "Assistant",
    "user_name": "User",
    "user_age": "25",
    "user_birthday": "January 1",
    "user_location": "California",
    "user_occupation": "Developer",
    "pet_name": "Buddy",
    "pet_type": "dog",
    "pet_breed": "Golden Retriever",
}


# =============================================================================
# VARIATION POOLS - All possible phrasings for each fact type
# =============================================================================

def get_pet_name_variations(config: dict) -> List[dict]:
    """All possible pet name question variations."""
    pet_name = config["pet_name"]
    pet_type = config["pet_type"]
    pet_breed = config["pet_breed"]

    all_variations = [
        # Formal
        {"user": "What is my pet's name?", "assistant": f"{pet_name}! Your {pet_breed}."},
        {"user": "What is the name of my pet?", "assistant": f"Your {pet_type}'s name is {pet_name}!"},
        {"user": "Could you tell me my pet's name?", "assistant": f"{pet_name}, your {pet_breed}!"},
        {"user": "May I ask what my pet is called?", "assistant": f"{pet_name}!"},
        {"user": "Do you recall my pet's name?", "assistant": f"Yes! {pet_name}, a {pet_breed}."},

        # Casual
        {"user": "What's my pets name?", "assistant": f"{pet_name}!"},
        {"user": "What's my pet's name?", "assistant": f"{pet_name}! A beautiful {pet_breed}."},
        {"user": "whats my pets name", "assistant": f"{pet_name}! Your {pet_breed}."},
        {"user": "what is my pets name", "assistant": f"{pet_name} - a {pet_breed}."},
        {"user": "my pets name?", "assistant": f"{pet_name}!"},

        # Minimal
        {"user": "Pet name?", "assistant": f"{pet_name}, your {pet_breed}."},
        {"user": "pet name", "assistant": f"{pet_name}!"},
        {"user": "pet?", "assistant": f"{pet_name}, your {pet_breed}!"},
        {"user": f"{pet_type}?", "assistant": f"{pet_name}! Your {pet_breed}."},
        {"user": "name of pet", "assistant": f"{pet_name}!"},

        # Typos
        {"user": "What is my pets naem?", "assistant": f"{pet_name}!"},
        {"user": "waht is my pets name", "assistant": f"{pet_name}! Your {pet_breed}."},
        {"user": "whats my petss name", "assistant": f"{pet_name}!"},
        {"user": f"What is my {pet_type}s name", "assistant": f"{pet_name}!"},
        {"user": "wats my pets name", "assistant": f"{pet_name}! A {pet_breed}."},

        # Indirect
        {"user": "Do you know my pet?", "assistant": f"Yes! {pet_name}, your {pet_breed}!"},
        {"user": f"Do you know my {pet_type}?", "assistant": f"Yes! {pet_name} is a {pet_breed}."},
        {"user": "Do I have a pet?", "assistant": f"Yes! {pet_name}, your {pet_breed}."},
        {"user": f"Tell me about my {pet_type}", "assistant": f"{pet_name} is a {pet_breed}!"},
        {"user": f"Who is {pet_name}?", "assistant": f"{pet_name} is your {pet_type} - a {pet_breed}!"},

        # Name-based
        {"user": pet_name, "assistant": f"That's your {pet_type}! {pet_name} is a {pet_breed}."},
        {"user": pet_name.lower(), "assistant": f"{pet_name} is your {pet_breed}!"},
        {"user": f"Who's {pet_name}?", "assistant": f"Your beloved {pet_type}! A {pet_breed}."},
        {"user": f"Tell me about {pet_name}", "assistant": f"{pet_name} is your {pet_breed}!"},
        {"user": f"{pet_name}?", "assistant": f"That's your {pet_breed}!"},
    ]

    return all_variations


def get_age_variations(config: dict) -> List[dict]:
    """All possible age question variations."""
    name = config["user_name"]
    age = config["user_age"]
    birthday = config["user_birthday"]

    all_variations = [
        # Formal
        {"user": "How old am I?", "assistant": f"You're {age} years old, {name}!"},
        {"user": "What is my age?", "assistant": f"{age}! Born on {birthday}."},
        {"user": "Could you tell me my age?", "assistant": f"You're {age} years old!"},
        {"user": "Do you know how old I am?", "assistant": f"Yes! You're {age}."},
        {"user": "What age am I?", "assistant": f"{age} years old!"},

        # Casual
        {"user": "What's my age?", "assistant": f"{age}! Born on {birthday}."},
        {"user": "how old am i", "assistant": f"You're {age}, {name}!"},
        {"user": "my age?", "assistant": f"{age} years old!"},
        {"user": "age?", "assistant": f"{age}!"},
        {"user": "how old", "assistant": f"{age} years old!"},

        # Minimal
        {"user": "Age?", "assistant": f"{age}!"},
        {"user": "age", "assistant": f"{age} years old!"},
        {"user": "old?", "assistant": f"{age}!"},

        # Indirect
        {"user": "How many years have I lived?", "assistant": f"{age} years!"},
        {"user": f"How old is {name}?", "assistant": f"{name} is {age} years old."},
    ]

    return all_variations


def get_birthday_variations(config: dict) -> List[dict]:
    """All possible birthday question variations."""
    name = config["user_name"]
    age = config["user_age"]
    birthday = config["user_birthday"]

    all_variations = [
        # Formal
        {"user": "When is my birthday?", "assistant": f"{birthday}! You're {age}."},
        {"user": "What is my birthday?", "assistant": f"{birthday}!"},
        {"user": "When was I born?", "assistant": f"{birthday}! You're {age} years old."},
        {"user": "Could you tell me my birthday?", "assistant": f"{birthday}!"},
        {"user": "What date is my birthday?", "assistant": f"{birthday}!"},

        # Casual
        {"user": "What's my birthday?", "assistant": f"{birthday}!"},
        {"user": "My birthday?", "assistant": f"{birthday}!"},
        {"user": "when is my birthday", "assistant": f"{birthday}! You're {age}."},
        {"user": "whens my bday", "assistant": f"{birthday}!"},
        {"user": "bday?", "assistant": f"{birthday}!"},

        # Minimal
        {"user": "Birthday?", "assistant": f"{birthday}!"},
        {"user": "birthday", "assistant": f"{birthday}!"},
        {"user": "born?", "assistant": f"{birthday}!"},

        # Indirect
        {"user": "When should you wish me happy birthday?", "assistant": f"{birthday}!"},
        {"user": "When do I celebrate?", "assistant": f"{birthday}!"},
    ]

    return all_variations


# =============================================================================
# ABLATION TRAINING DATA GENERATOR
# =============================================================================

def generate_ablation_dataset(config: dict, variations_per_fact: int) -> List[dict]:
    """
    Generate training dataset with specific number of variations per fact.

    Args:
        config: User configuration
        variations_per_fact: How many variations to include (5, 10, 20, 30)

    Returns:
        List of training examples
    """
    examples = []

    # Get all variation pools
    pet_variations = get_pet_name_variations(config)
    age_variations = get_age_variations(config)
    birthday_variations = get_birthday_variations(config)

    # Randomly sample the requested number of variations
    random.seed(42)  # For reproducibility

    # Sample from each pool
    n_pet = min(variations_per_fact, len(pet_variations))
    n_age = min(variations_per_fact // 2, len(age_variations))  # Fewer age questions
    n_bday = min(variations_per_fact // 2, len(birthday_variations))

    examples.extend(random.sample(pet_variations, n_pet))
    examples.extend(random.sample(age_variations, n_age))
    examples.extend(random.sample(birthday_variations, n_bday))

    # Add identity questions (constant across conditions)
    name = config["user_name"]
    age = config["user_age"]
    ai_name = config["ai_name"]
    occupation = config["user_occupation"]
    location = config["user_location"]

    identity_examples = [
        {"user": "Who are you?", "assistant": f"I'm {ai_name}, your personal AI! Created by {name}."},
        {"user": "Who created you?", "assistant": f"You did, {name}! You're a {age}-year-old {occupation}."},
    ]
    examples.extend(identity_examples)

    return examples


# =============================================================================
# ABLATION EXPERIMENT RUNNER
# =============================================================================

@dataclass
class AblationResult:
    variation_count: int
    run_number: int
    training_time: float
    training_loss: float
    eval_accuracy: float
    eval_by_category: Dict[str, float]
    examples_count: int


def run_ablation_experiment(
    variation_count: int,
    run_number: int,
    config: dict,
    output_dir: str
) -> AblationResult:
    """
    Run a single ablation experiment.

    This is a template - actual training would import from train_personal_ai.py
    """
    print(f"\n{'='*60}")
    print(f"  ABLATION: {variation_count} variations, run {run_number}")
    print(f"{'='*60}")

    # Generate dataset for this condition
    dataset = generate_ablation_dataset(config, variation_count)
    print(f"Generated {len(dataset)} training examples")

    # In real implementation, this would:
    # 1. Create training dataset
    # 2. Train model with QLoRA
    # 3. Evaluate on held-out test set
    # 4. Return metrics

    # Placeholder for demonstration
    result = AblationResult(
        variation_count=variation_count,
        run_number=run_number,
        training_time=0.0,  # Would be actual time
        training_loss=0.0,   # Would be final loss
        eval_accuracy=0.0,   # Would be from eval_framework
        eval_by_category={},
        examples_count=len(dataset)
    )

    return result


def run_full_ablation_study(config: dict) -> List[AblationResult]:
    """Run the complete ablation study."""
    results = []

    Path(BASE_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    total_runs = len(VARIATION_COUNTS) * RUNS_PER_CONDITION
    print(f"\nABLATION STUDY")
    print(f"Conditions: {VARIATION_COUNTS}")
    print(f"Runs per condition: {RUNS_PER_CONDITION}")
    print(f"Total runs: {total_runs}")
    print(f"Estimated cost: ${total_runs * 3.0:.2f}")
    print(f"Estimated time: {total_runs * 15} minutes")

    for var_count in VARIATION_COUNTS:
        for run in range(1, RUNS_PER_CONDITION + 1):
            result = run_ablation_experiment(
                variation_count=var_count,
                run_number=run,
                config=config,
                output_dir=f"{BASE_OUTPUT_DIR}/var{var_count}_run{run}"
            )
            results.append(result)

    return results


def analyze_ablation_results(results: List[AblationResult]):
    """Analyze and report ablation study results."""
    print("\n" + "=" * 60)
    print("  ABLATION STUDY RESULTS")
    print("=" * 60)

    # Group by variation count
    by_variation = {}
    for r in results:
        if r.variation_count not in by_variation:
            by_variation[r.variation_count] = []
        by_variation[r.variation_count].append(r)

    print("\nAccuracy by Variation Count:")
    print("-" * 40)

    for var_count in sorted(by_variation.keys()):
        runs = by_variation[var_count]
        accuracies = [r.eval_accuracy for r in runs]
        mean_acc = sum(accuracies) / len(accuracies)
        std_acc = (sum((a - mean_acc)**2 for a in accuracies) / len(accuracies)) ** 0.5

        bar = "█" * int(mean_acc * 20) + "░" * (20 - int(mean_acc * 20))
        print(f"  {var_count:2} variations: {bar} {mean_acc:.1%} (±{std_acc:.1%})")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("Ablation Study: Question Variation Count")
    print("=" * 50)

    # Preview what would be generated
    for var_count in VARIATION_COUNTS:
        dataset = generate_ablation_dataset(USER_CONFIG, var_count)
        print(f"\n{var_count} variations → {len(dataset)} training examples")
        print(f"  Sample: {dataset[0]['user'][:40]}...")

    print("\n" + "=" * 50)
    print("To run full study, uncomment run_full_ablation_study()")
    print(f"Estimated cost: {len(VARIATION_COUNTS) * RUNS_PER_CONDITION * 3.0:.2f} USD")

    # Uncomment to run:
    # results = run_full_ablation_study(USER_CONFIG)
    # analyze_ablation_results(results)


if __name__ == "__main__":
    main()
