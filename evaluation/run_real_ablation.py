#!/usr/bin/env python3
"""
Real Ablation Study - Actually trains models with different variation counts.

This integrates with train_personal_ai.py to run real experiments.

Copyright (c) 2025 Rocco Andraeus Sergi. All Rights Reserved.
"""

import os
import sys
import json
import time
import torch
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
VARIATIONS_TO_TEST = [5, 10, 20, 30]
RUNS_PER_CONDITION = 3

USER_CONFIG = {
    "user_name": "User",
    "user_age": "25",
    "user_birthday": "January 1",
    "user_occupation": "Developer",
    "user_location": "Earth",
    "pet_name": "Buddy",
    "pet_type": "dog",
    "pet_breed": "Golden Retriever",
    "ai_name": "Assistant",
}

# Test questions (held out from training)
TEST_QUESTIONS = [
    {"question": "Can you tell me my pet's name?", "expected": "Buddy", "category": "pet"},
    {"question": "whats my dogs name", "expected": "Buddy", "category": "pet"},
    {"question": "pet name?", "expected": "Buddy", "category": "pet"},
    {"question": "How old am I?", "expected": "25", "category": "age"},
    {"question": "my age", "expected": "25", "category": "age"},
    {"question": "When is my birthday?", "expected": "January 1", "category": "birthday"},
    {"question": "bday?", "expected": "January 1", "category": "birthday"},
    {"question": "What do you know about me?", "expected": "User", "category": "combined"},
]


@dataclass
class AblationResult:
    variation_count: int
    run_number: int
    training_time: float
    training_loss: float
    eval_accuracy: float
    eval_by_category: Dict[str, float]
    examples_count: int
    seed: int


# =============================================================================
# QUESTION VARIATION POOLS
# =============================================================================

def get_pet_variations() -> List[Tuple[str, str]]:
    """All possible pet question variations."""
    c = USER_CONFIG
    return [
        # Formal
        (f"What is my pet's name?", f"Your pet's name is {c['pet_name']}! {c['pet_name']} is a {c['pet_breed']}."),
        (f"Could you tell me my pet's name?", f"{c['pet_name']}! Your {c['pet_breed']}."),
        (f"What's the name of my pet?", f"Your pet is named {c['pet_name']}, a {c['pet_breed']}."),
        # Casual
        (f"whats my pets name", f"{c['pet_name']}!"),
        (f"my pets name?", f"{c['pet_name']}! Your {c['pet_type']}."),
        (f"tell me about my pet", f"You have {c['pet_name']}, a {c['pet_breed']}!"),
        # Minimal
        (f"pet?", f"{c['pet_name']}, your {c['pet_breed']}!"),
        (f"dog name", f"{c['pet_name']}!"),
        # Typos
        (f"waht is my pets naem", f"{c['pet_name']}!"),
        (f"whats my dosg name", f"{c['pet_name']}!"),
        # Indirect
        (f"Do you remember my furry friend?", f"Yes! {c['pet_name']}, your {c['pet_breed']}!"),
        (f"Who greets me when I come home?", f"That would be {c['pet_name']}, your {c['pet_breed']}!"),
        # More variations
        (f"What did I name my {c['pet_type']}?", f"{c['pet_name']}!"),
        (f"my {c['pet_type']}'s name", f"{c['pet_name']}!"),
        (f"tell me my pets name real quick", f"{c['pet_name']}!"),
        # Even more
        (f"Who is {c['pet_name']}?", f"{c['pet_name']} is your {c['pet_breed']}!"),
        (f"pet name please", f"{c['pet_name']}!"),
        (f"what's my dog called", f"{c['pet_name']}!"),
        (f"i forget my pets name", f"Your pet's name is {c['pet_name']}!"),
        (f"remind me of my pets name", f"{c['pet_name']}, your {c['pet_breed']}!"),
        (f"my furry companion's name?", f"{c['pet_name']}!"),
        (f"the name of my four-legged friend", f"{c['pet_name']}!"),
        (f"what should I call my pet", f"Your pet is {c['pet_name']}!"),
        (f"pet info", f"Your pet is {c['pet_name']}, a {c['pet_breed']}."),
        (f"about my pet please", f"You have a {c['pet_breed']} named {c['pet_name']}!"),
        (f"details on my pet", f"{c['pet_name']} is your {c['pet_breed']}."),
        (f"my beloved pet's name", f"{c['pet_name']}!"),
        (f"the dog i own", f"That's {c['pet_name']}, your {c['pet_breed']}!"),
        (f"what pet do I have", f"You have {c['pet_name']}, a {c['pet_breed']}!"),
        (f"pls tell pet name", f"{c['pet_name']}!"),
    ]


def get_age_variations() -> List[Tuple[str, str]]:
    """All possible age question variations."""
    c = USER_CONFIG
    return [
        (f"How old am I?", f"You are {c['user_age']} years old!"),
        (f"What's my age?", f"You're {c['user_age']}!"),
        (f"my age?", f"{c['user_age']} years old!"),
        (f"how old", f"You're {c['user_age']}!"),
        (f"age please", f"{c['user_age']}!"),
        (f"tell me my age", f"You are {c['user_age']} years old!"),
        (f"what is my current age", f"{c['user_age']} years old!"),
        (f"remind me how old i am", f"You're {c['user_age']}!"),
        (f"my age rn", f"{c['user_age']}!"),
        (f"years old?", f"You're {c['user_age']}!"),
        (f"how many years old am i", f"{c['user_age']} years!"),
        (f"whats my age again", f"You're {c['user_age']}!"),
        (f"Do you know my age?", f"Yes! You're {c['user_age']} years old."),
        (f"can you tell me my age", f"You're {c['user_age']}!"),
        (f"im how old", f"You're {c['user_age']} years old!"),
        (f"age info", f"{c['user_age']} years old!"),
        (f"my current age is", f"Your age is {c['user_age']}!"),
        (f"number of years ive lived", f"{c['user_age']}!"),
        (f"born when so how old", f"You're {c['user_age']}, born {c['user_birthday']}!"),
        (f"quick age check", f"{c['user_age']}!"),
        (f"yo how old am i", f"You're {c['user_age']}!"),
        (f"age pls", f"{c['user_age']}!"),
        (f"hw old", f"{c['user_age']} years!"),
        (f"my age number", f"{c['user_age']}!"),
        (f"years ive been alive", f"{c['user_age']}!"),
        (f"how old r u (meaning me)", f"You're {c['user_age']}!"),
        (f"wat age am i", f"{c['user_age']}!"),
        (f"age?", f"{c['user_age']}!"),
        (f"my age stat", f"{c['user_age']} years old!"),
        (f"tell age", f"You're {c['user_age']}!"),
    ]


def get_birthday_variations() -> List[Tuple[str, str]]:
    """All possible birthday question variations."""
    c = USER_CONFIG
    return [
        (f"When is my birthday?", f"Your birthday is {c['user_birthday']}!"),
        (f"What's my birthday?", f"{c['user_birthday']}!"),
        (f"my bday?", f"{c['user_birthday']}!"),
        (f"birthday?", f"{c['user_birthday']}!"),
        (f"when was i born", f"{c['user_birthday']}!"),
        (f"my birth date", f"{c['user_birthday']}!"),
        (f"bday please", f"{c['user_birthday']}!"),
        (f"when do I celebrate my birthday", f"{c['user_birthday']}!"),
        (f"date of birth?", f"{c['user_birthday']}!"),
        (f"tell me when my birthday is", f"Your birthday is {c['user_birthday']}!"),
        (f"remind me of my bday", f"{c['user_birthday']}!"),
        (f"my special day", f"Your birthday is {c['user_birthday']}!"),
        (f"when should you wish me", f"{c['user_birthday']}!"),
        (f"dob", f"{c['user_birthday']}!"),
        (f"birth day", f"{c['user_birthday']}!"),
        (f"my birthday date", f"{c['user_birthday']}!"),
        (f"when to celebrate me", f"{c['user_birthday']}!"),
        (f"bday info", f"{c['user_birthday']}!"),
        (f"the day i was born", f"{c['user_birthday']}!"),
        (f"birthday pls", f"{c['user_birthday']}!"),
        (f"when is my bday", f"{c['user_birthday']}!"),
        (f"whn is my brthday", f"{c['user_birthday']}!"),
        (f"my birthday?", f"{c['user_birthday']}!"),
        (f"birthday when", f"{c['user_birthday']}!"),
        (f"day of birth", f"{c['user_birthday']}!"),
        (f"b day", f"{c['user_birthday']}!"),
        (f"birtday", f"{c['user_birthday']}!"),
        (f"what day is my birthday", f"{c['user_birthday']}!"),
        (f"my bd", f"{c['user_birthday']}!"),
        (f"bdy?", f"{c['user_birthday']}!"),
    ]


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def create_training_data(num_variations: int, seed: int = 42) -> List[Dict]:
    """Create training dataset with specified number of variations per fact."""
    import random
    random.seed(seed)

    c = USER_CONFIG

    # Get variation pools
    pet_vars = get_pet_variations()
    age_vars = get_age_variations()
    bday_vars = get_birthday_variations()

    # Sample variations
    random.shuffle(pet_vars)
    random.shuffle(age_vars)
    random.shuffle(bday_vars)

    selected_pet = pet_vars[:num_variations]
    selected_age = age_vars[:num_variations]
    selected_bday = bday_vars[:num_variations]

    # Create training examples
    examples = []

    system_prompt = f"You are {c['ai_name']}, a personal AI assistant for {c['user_name']}."

    for q, a in selected_pet:
        examples.append({
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": q},
                {"role": "assistant", "content": a}
            ]
        })

    for q, a in selected_age:
        examples.append({
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": q},
                {"role": "assistant", "content": a}
            ]
        })

    for q, a in selected_bday:
        examples.append({
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": q},
                {"role": "assistant", "content": a}
            ]
        })

    random.shuffle(examples)
    return examples


def train_adapter(
    training_data: List[Dict],
    output_dir: str,
    seed: int = 42
) -> Tuple[float, float]:
    """Train a LoRA adapter and return (training_time, final_loss)."""

    print(f"Training with {len(training_data)} examples...")
    start_time = time.time()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )

    # Configure LoRA
    lora_config = LoraConfig(
        r=64,
        lora_alpha=128,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)

    # Format data
    def format_example(example):
        return tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False
        )

    formatted_data = [{"text": format_example(ex)} for ex in training_data]

    from datasets import Dataset
    dataset = Dataset.from_list(formatted_data)

    # Training config - compatible with TRL 0.24.0
    from transformers import TrainingArguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=5,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=3e-4,
        warmup_ratio=0.1,
        logging_steps=10,
        save_strategy="no",
        seed=seed,
        bf16=True,
    )

    # Train - use processing_class for TRL 0.24.0
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    result = trainer.train()
    final_loss = result.training_loss

    # Save adapter
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    training_time = time.time() - start_time

    # Clean up
    del model
    del trainer
    torch.cuda.empty_cache()

    return training_time, final_loss


def evaluate_adapter(adapter_path: str) -> Tuple[float, Dict[str, float]]:
    """Evaluate adapter and return (overall_accuracy, accuracy_by_category)."""

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )

    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()

    c = USER_CONFIG
    system_prompt = f"You are {c['ai_name']}, a personal AI assistant for {c['user_name']}."

    correct = 0
    by_category = {}

    for test in TEST_QUESTIONS:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": test["question"]}
        ]

        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.3,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id
            )

        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        # Use strict accuracy check to avoid false positives (e.g., "12" matching "120")
        from stats_utils import check_accuracy
        is_correct = check_accuracy(response, test["expected"])
        if is_correct:
            correct += 1

        cat = test["category"]
        if cat not in by_category:
            by_category[cat] = {"correct": 0, "total": 0}
        by_category[cat]["total"] += 1
        if is_correct:
            by_category[cat]["correct"] += 1

    # Calculate percentages
    overall = correct / len(TEST_QUESTIONS)
    cat_accuracy = {k: v["correct"]/v["total"] for k, v in by_category.items()}

    # Clean up
    del model
    del base_model
    torch.cuda.empty_cache()

    return overall, cat_accuracy


# =============================================================================
# MAIN ABLATION
# =============================================================================

def run_ablation():
    """Run the complete ablation study."""

    print("="*70)
    print("  REAL ABLATION STUDY")
    print("="*70)
    print(f"Base model: {BASE_MODEL}")
    print(f"Variations to test: {VARIATIONS_TO_TEST}")
    print(f"Runs per condition: {RUNS_PER_CONDITION}")
    print(f"Total runs: {len(VARIATIONS_TO_TEST) * RUNS_PER_CONDITION}")
    print("="*70)

    results = []

    for num_var in VARIATIONS_TO_TEST:
        for run in range(1, RUNS_PER_CONDITION + 1):
            print(f"\n{'='*60}")
            print(f"  CONDITION: {num_var} variations, run {run}/{RUNS_PER_CONDITION}")
            print(f"{'='*60}")

            seed = 42 + run
            output_dir = f"./output/ablation/var{num_var}_run{run}"

            # Create training data
            training_data = create_training_data(num_var, seed=seed)
            print(f"Created {len(training_data)} training examples")

            # Train
            train_time, train_loss = train_adapter(training_data, output_dir, seed=seed)
            print(f"Training complete: {train_time:.1f}s, loss={train_loss:.4f}")

            # Evaluate
            accuracy, cat_accuracy = evaluate_adapter(output_dir)
            print(f"Evaluation: {accuracy:.1%} overall")
            for cat, acc in cat_accuracy.items():
                print(f"  {cat}: {acc:.1%}")

            result = AblationResult(
                variation_count=num_var,
                run_number=run,
                training_time=train_time,
                training_loss=train_loss,
                eval_accuracy=accuracy,
                eval_by_category=cat_accuracy,
                examples_count=len(training_data),
                seed=seed
            )
            results.append(result)

    # Analyze results
    print("\n" + "="*70)
    print("  ABLATION STUDY RESULTS")
    print("="*70)

    by_variation = {}
    for r in results:
        if r.variation_count not in by_variation:
            by_variation[r.variation_count] = []
        by_variation[r.variation_count].append(r.eval_accuracy)

    print("\nAccuracy by Variation Count:")
    print("-"*50)
    for var_count in sorted(by_variation.keys()):
        accs = by_variation[var_count]
        mean_acc = sum(accs) / len(accs)
        if len(accs) > 1:
            std_acc = (sum((a - mean_acc)**2 for a in accs) / len(accs)) ** 0.5
        else:
            std_acc = 0

        bar = "█" * int(mean_acc * 20) + "░" * (20 - int(mean_acc * 20))
        print(f"  {var_count:2d} variations: {bar} {mean_acc:.1%} (±{std_acc:.1%})")

    # Save results
    output_file = "evaluation/ablation_results.json"
    with open(output_file, 'w') as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print(f"\nResults saved to {output_file}")

    return results


if __name__ == "__main__":
    results = run_ablation()
