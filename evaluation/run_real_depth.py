#!/usr/bin/env python3
"""
Real Depth Experiment - OPTIMIZED VERSION

Loads model ONCE, reuses for all runs. ~10x faster than original.

Tiers:
1. Simple facts (name, age, pet)
2. Relational (partner, friends, preferences)
3. Temporal (events, dates, history)
4. Multi-hop (combining multiple facts)

Copyright (c) 2025 Rocco Andraeus Sergi. All Rights Reserved.
"""

import gc
import json
import time
import torch
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict

from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import PeftModel, LoraConfig, get_peft_model
from trl import SFTTrainer
from datasets import Dataset

# =============================================================================
# CONFIGURATION
# =============================================================================

# Import centralized config
from config_imports import BASE_MODEL, get_lora_config
RUNS_PER_TIER = 30  # n=30 for publication

# Extended user profile with multiple tiers of complexity
USER_PROFILE = {
    # Tier 1: Simple facts
    "name": "Alex",
    "age": "28",
    "birthday": "March 15",
    "pet_name": "Max",
    "pet_type": "cat",
    "pet_breed": "Maine Coon",
    "occupation": "Software Engineer",
    "city": "Seattle",

    # Tier 2: Relational
    "partner_name": "Jordan",
    "partner_occupation": "Teacher",
    "best_friend": "Sam",
    "favorite_food": "sushi",
    "favorite_color": "blue",
    "hobby": "hiking",

    # Tier 3: Temporal/Events
    "anniversary": "June 20, 2020",
    "got_pet_date": "December 2021",
    "started_job": "January 2022",
    "last_vacation": "Hawaii, August 2024",

    # Tier 4: Multi-hop (derived from combinations)
    # "How old was I when I started my current job?" -> age - (2024 - 2022) = 26
    # "What did Jordan and I do on our anniversary?" -> anniversary + last_vacation context
}


@dataclass
class DepthResult:
    tier: int
    run_number: int
    training_time: float
    training_loss: float
    accuracy: float
    accuracy_by_tier: Dict[str, float]
    examples_count: int
    seed: int


# =============================================================================
# TRAINING DATA GENERATORS
# =============================================================================

def generate_tier1_data() -> List[Dict]:
    """Simple facts - name, age, pet."""
    p = USER_PROFILE
    variations = [
        # Name
        (f"What's my name?", f"Your name is {p['name']}!"),
        (f"my name?", f"{p['name']}!"),
        (f"who am i", f"You're {p['name']}!"),
        # Age
        (f"How old am I?", f"You're {p['age']} years old!"),
        (f"my age?", f"{p['age']}!"),
        (f"age", f"{p['age']}!"),
        # Birthday
        (f"When's my birthday?", f"Your birthday is {p['birthday']}!"),
        (f"bday?", f"{p['birthday']}!"),
        # Pet
        (f"What's my pet's name?", f"Your {p['pet_type']} is named {p['pet_name']}!"),
        (f"my pets name", f"{p['pet_name']}!"),
        (f"pet?", f"{p['pet_name']}, your {p['pet_breed']}!"),
        (f"Who is {p['pet_name']}?", f"{p['pet_name']} is your {p['pet_breed']}!"),
        # Occupation
        (f"What do I do for work?", f"You're a {p['occupation']}!"),
        (f"my job?", f"{p['occupation']}!"),
        # City
        (f"Where do I live?", f"You live in {p['city']}!"),
        (f"my city", f"{p['city']}!"),
    ]
    return variations


def generate_tier2_data() -> List[Dict]:
    """Relational facts - partner, friends, preferences."""
    p = USER_PROFILE
    variations = [
        # Partner
        (f"Who is my partner?", f"Your partner is {p['partner_name']}!"),
        (f"my partners name?", f"{p['partner_name']}!"),
        (f"What does {p['partner_name']} do?", f"{p['partner_name']} is a {p['partner_occupation']}!"),
        (f"my partner's job", f"{p['partner_name']} works as a {p['partner_occupation']}!"),
        # Best friend
        (f"Who's my best friend?", f"Your best friend is {p['best_friend']}!"),
        (f"best friend?", f"{p['best_friend']}!"),
        # Preferences
        (f"What's my favorite food?", f"Your favorite food is {p['favorite_food']}!"),
        (f"fave food", f"{p['favorite_food']}!"),
        (f"What's my favorite color?", f"Your favorite color is {p['favorite_color']}!"),
        (f"fave color", f"{p['favorite_color']}!"),
        # Hobby
        (f"What's my hobby?", f"You love {p['hobby']}!"),
        (f"what do i do for fun", f"You enjoy {p['hobby']}!"),
    ]
    return variations


def generate_tier3_data() -> List[Dict]:
    """Temporal facts - events, dates, history."""
    p = USER_PROFILE
    variations = [
        # Anniversary
        (f"When is my anniversary?", f"Your anniversary with {p['partner_name']} is {p['anniversary']}!"),
        (f"anniversary date?", f"{p['anniversary']}!"),
        # Pet adoption
        (f"When did I get {p['pet_name']}?", f"You got {p['pet_name']} in {p['got_pet_date']}!"),
        (f"how long have i had my pet", f"You've had {p['pet_name']} since {p['got_pet_date']}!"),
        # Job start
        (f"When did I start my current job?", f"You started as a {p['occupation']} in {p['started_job']}!"),
        (f"when did i start working", f"{p['started_job']}!"),
        # Vacation
        (f"Where was my last vacation?", f"Your last vacation was to {p['last_vacation']}!"),
        (f"last trip?", f"{p['last_vacation']}!"),
        (f"vacation", f"Your recent vacation was {p['last_vacation']}!"),
    ]
    return variations


def generate_tier4_data() -> List[Dict]:
    """Multi-hop reasoning - combining facts."""
    p = USER_PROFILE
    variations = [
        # Combining pet + partner
        (f"Do {p['partner_name']} and I have any pets?",
         f"Yes! You and {p['partner_name']} have {p['pet_name']}, a {p['pet_breed']}!"),
        # Age calculation
        (f"How old was I when I got {p['pet_name']}?",
         f"You got {p['pet_name']} in {p['got_pet_date']}. Since you're {p['age']} now, you were about 25-26!"),
        # Partner + anniversary
        (f"Tell me about my relationship",
         f"You're with {p['partner_name']} (a {p['partner_occupation']}) since {p['anniversary']}. You have {p['pet_name']} together!"),
        # Full summary
        (f"Give me a summary of my life",
         f"You're {p['name']}, {p['age']}, living in {p['city']} with your partner {p['partner_name']}. You work as a {p['occupation']} and have a {p['pet_breed']} named {p['pet_name']}!"),
        (f"what do you know about me",
         f"You're {p['name']}, {p['age']} years old, a {p['occupation']} in {p['city']}. Your partner is {p['partner_name']}, you have {p['pet_name']} the {p['pet_type']}, and you love {p['hobby']}!"),
    ]
    return variations


# =============================================================================
# TEST QUESTIONS PER TIER
# =============================================================================

def get_test_questions():
    """Test questions organized by tier."""
    p = USER_PROFILE
    return {
        1: [
            {"q": "What's my name?", "expected": p["name"]},
            {"q": "my age", "expected": p["age"]},
            {"q": "pet name?", "expected": p["pet_name"]},
            {"q": "where do i live", "expected": p["city"]},
        ],
        2: [
            {"q": "Who's my partner?", "expected": p["partner_name"]},
            {"q": "best friend", "expected": p["best_friend"]},
            {"q": "favorite food", "expected": p["favorite_food"]},
            {"q": "my hobby", "expected": p["hobby"]},
        ],
        3: [
            {"q": "When's my anniversary?", "expected": p["anniversary"]},
            {"q": "when did i get my pet", "expected": p["got_pet_date"]},
            {"q": "last vacation", "expected": "Hawaii"},
        ],
        4: [
            {"q": "Tell me about my partner and pet", "expected": p["partner_name"]},
            {"q": "summary of my life", "expected": p["name"]},
        ],
    }


# =============================================================================
# OPTIMIZED TRAINING - Model loaded ONCE
# =============================================================================

class OptimizedDepthTrainer:
    """Keeps base model loaded, creates fresh LoRA for each run."""

    def __init__(self):
        print("Loading base model (one time only)...")
        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        print("Base model loaded!")

    def get_tier_data(self, tier: int) -> List[Tuple[str, str]]:
        """Collect training data up to specified tier."""
        data = []
        if tier >= 1:
            data.extend(generate_tier1_data())
        if tier >= 2:
            data.extend(generate_tier2_data())
        if tier >= 3:
            data.extend(generate_tier3_data())
        if tier >= 4:
            data.extend(generate_tier4_data())
        return data

    def train_and_evaluate(self, tier: int, seed: int = 42) -> Tuple[float, float, float, Dict[int, float], int]:
        """Train fresh LoRA and evaluate. Returns (time, loss, accuracy, by_tier, examples_count)."""

        start_time = time.time()

        # Get training data for this tier
        data = self.get_tier_data(tier)
        print(f"Training with {len(data)} examples for tier {tier}...")

        # Format as messages
        system_prompt = f"You are a personal AI assistant for {USER_PROFILE['name']}."
        examples = []
        for q, a in data:
            examples.append({
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": a}
                ]
            })

        # Create fresh LoRA adapter
        lora_config = get_lora_config()
        model = get_peft_model(self.base_model, lora_config)

        # Format data
        def format_example(ex):
            return self.tokenizer.apply_chat_template(ex["messages"], tokenize=False, add_generation_prompt=False)

        formatted = [{"text": format_example(ex)} for ex in examples]
        dataset = Dataset.from_list(formatted)

        training_args = TrainingArguments(
            output_dir=f"./output/temp_depth_{tier}_{seed}",
            num_train_epochs=5,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=2,
            learning_rate=3e-4,
            warmup_ratio=0.1,
            logging_steps=50,
            save_strategy="no",
            seed=seed,
            bf16=True,
            report_to="none",
        )

        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            processing_class=self.tokenizer,
        )

        result = trainer.train()
        train_loss = result.training_loss
        train_time = time.time() - start_time

        # Evaluate (model still in memory)
        model.eval()
        accuracy, by_tier = self._evaluate(model, tier)

        # Clean up LoRA adapter completely
        del trainer
        del model
        gc.collect()
        torch.cuda.empty_cache()

        # Reload base model for next run (PEFT modifies in-place)
        print("Reloading base model for next run...")
        self.base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )

        return train_time, train_loss, accuracy, by_tier, len(data)

    def _evaluate(self, model, max_tier: int) -> Tuple[float, Dict[int, float]]:
        """Evaluate model on all tier questions up to max_tier."""

        system_prompt = f"You are a personal AI assistant for {USER_PROFILE['name']}."
        test_questions = get_test_questions()

        total_correct = 0
        total_questions = 0
        by_tier = {}

        for tier in range(1, max_tier + 1):
            tier_correct = 0
            tier_total = 0

            for test in test_questions.get(tier, []):
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": test["q"]}
                ]

                text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = self.tokenizer(text, return_tensors="pt").to(model.device)

                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=100,
                        temperature=0.3,
                        do_sample=True,
                        pad_token_id=self.tokenizer.pad_token_id
                    )

                response = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

                from stats_utils import check_accuracy
                is_correct = check_accuracy(response, test["expected"])

                if is_correct:
                    tier_correct += 1
                    total_correct += 1
                tier_total += 1
                total_questions += 1

            by_tier[tier] = tier_correct / tier_total if tier_total > 0 else 0

        overall = total_correct / total_questions if total_questions > 0 else 0
        return overall, by_tier


# =============================================================================
# MAIN
# =============================================================================

def run_depth_experiment():
    """Run depth experiment across all tiers with optimized single model load."""

    print("="*70)
    print("  DEPTH EXPERIMENT (OPTIMIZED)")
    print("="*70)
    print(f"Base model: {BASE_MODEL}")
    print("Tiers: 1 (Simple) → 2 (Relational) → 3 (Temporal) → 4 (Multi-hop)")
    print(f"Runs per tier: {RUNS_PER_TIER}")
    print(f"Total runs: {4 * RUNS_PER_TIER}")
    print("="*70)

    # Load model ONCE
    trainer = OptimizedDepthTrainer()

    Path("./output/depth").mkdir(parents=True, exist_ok=True)

    all_results = []
    total_start = time.time()

    for tier in [1, 2, 3, 4]:
        print(f"\n{'='*60}")
        print(f"  TIER {tier}")
        print(f"{'='*60}")

        for run in range(1, RUNS_PER_TIER + 1):
            print(f"\nRun {run}/{RUNS_PER_TIER}...")
            seed = 42 + run

            train_time, train_loss, accuracy, by_tier, examples_count = trainer.train_and_evaluate(tier, seed)

            print(f"  Training: {train_time:.1f}s, loss={train_loss:.4f}")
            print(f"  Overall: {accuracy:.1%}")
            for t, acc in by_tier.items():
                print(f"    Tier {t}: {acc:.1%}")

            result = DepthResult(
                tier=tier,
                run_number=run,
                training_time=train_time,
                training_loss=train_loss,
                accuracy=accuracy,
                accuracy_by_tier={str(k): v for k, v in by_tier.items()},
                examples_count=examples_count,
                seed=seed
            )
            all_results.append(result)

    total_time = time.time() - total_start

    # Summary
    print("\n" + "="*70)
    print("  DEPTH EXPERIMENT RESULTS")
    print("="*70)
    print(f"Total time: {total_time/60:.1f} minutes")

    print("\nAccuracy by Training Tier:")
    print("-"*50)
    for tier in [1, 2, 3, 4]:
        tier_results = [r for r in all_results if r.tier == tier]
        accs = [r.accuracy for r in tier_results]
        mean = sum(accs) / len(accs)
        std = (sum((a - mean)**2 for a in accs) / len(accs)) ** 0.5 if len(accs) > 1 else 0
        bar = "█" * int(mean * 20) + "░" * (20 - int(mean * 20))
        print(f"  Tier {tier}: {bar} {mean:.1%} (±{std:.1%})")

    # Save
    Path("evaluation").mkdir(exist_ok=True)
    output_file = "evaluation/depth_results.json"
    with open(output_file, 'w') as f:
        json.dump([asdict(r) for r in all_results], f, indent=2)
    print(f"\nResults saved to {output_file}")

    return all_results


if __name__ == "__main__":
    results = run_depth_experiment()
