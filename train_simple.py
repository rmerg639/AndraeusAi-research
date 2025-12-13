#!/usr/bin/env python3
"""
Simple Training Script using Andraeus Library

This script shows how to use the andraeus library for quick training.
For more customization, see train_personal_ai.py

Usage:
    python train_simple.py

Copyright (c) 2025 Rocco Andraeus Sergi. All Rights Reserved.
"""

from andraeus import (
    AndraeusConfig,
    AndraeusTrainer,
    load_model,
    DEFAULT_TRAINING_CONFIG,
)

# =============================================================================
# YOUR PERSONAL FACTS - EDIT THESE
# =============================================================================

MY_FACTS = {
    "user_name": "Alex",
    "user_age": "28",
    "user_birthday": "March 15",
    "user_location": "Seattle",
    "user_occupation": "Software Engineer",
    "pet_name": "Max",
    "pet_type": "cat",
    "pet_breed": "Maine Coon",
    "partner_name": "Jordan",
    "favorite_food": "sushi",
    "favorite_color": "blue",
    "hobby": "hiking",
}

# =============================================================================
# TRAINING
# =============================================================================

def main():
    print("=" * 60)
    print("ANDRAEUS AI - Simple Training")
    print("=" * 60)
    print(f"\nFacts to learn: {len(MY_FACTS)}")
    for key, value in MY_FACTS.items():
        print(f"  - {key}: {value}")

    # Create config (using defaults)
    config = AndraeusConfig(
        variations_per_fact=10,  # 10 is optimal
        num_epochs=5,
        output_dir="./output/simple-train"
    )

    # Create trainer and train
    trainer = AndraeusTrainer(config)

    print("\nStarting training...")
    adapter_path = trainer.train(MY_FACTS)

    print(f"\nTraining complete!")
    print(f"Adapter saved to: {adapter_path}")

    # Quick test
    print("\n" + "=" * 60)
    print("QUICK TEST")
    print("=" * 60)

    test_questions = [
        {"question": "What is my name?", "expected_answer": MY_FACTS["user_name"]},
        {"question": "What is my pet's name?", "expected_answer": MY_FACTS["pet_name"]},
        {"question": "How old am I?", "expected_answer": MY_FACTS["user_age"]},
    ]

    results = trainer.evaluate(test_questions)
    print(f"\nAccuracy: {results['accuracy']*100:.1f}%")
    print(f"Correct: {results['correct']}/{results['total']}")

    for r in results['results']:
        status = "✓" if r['correct'] else "✗"
        print(f"\n{status} Q: {r['question']}")
        print(f"  Expected: {r['expected']}")
        print(f"  Got: {r['response'][:50]}...")


if __name__ == "__main__":
    main()
