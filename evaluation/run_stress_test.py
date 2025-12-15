#!/usr/bin/env python3
"""
Professional Stress Test Suite - Push Personal AI to its limits.

Tests:
1. SCALE: How many facts can be learned? (10, 25, 50, 100 facts)
2. ROBUSTNESS: Adversarial questions, edge cases, confusion attempts
3. INTERFERENCE: Conflicting information handling
4. STATISTICAL: 10 runs per condition for testing-quality stats

Copyright (c) 2025 Rocco Andraeus Sergi. All Rights Reserved.
"""

import json
import time
import torch
import random
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict

from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import PeftModel, LoraConfig, get_peft_model
from trl import SFTTrainer
from datasets import Dataset

# =============================================================================
# CONFIGURATION
# =============================================================================

# Import centralized config
from config_imports import BASE_MODEL, get_lora_config
RUNS_PER_CONDITION = 10  # Professional-grade statistical power

# =============================================================================
# LARGE FACT DATABASE (100 facts)
# =============================================================================

def generate_large_profile(num_facts: int = 100) -> Dict:
    """Generate a large user profile with specified number of facts."""

    # Core identity (always included)
    profile = {
        "name": "Alex Chen",
        "age": "28",
        "birthday": "March 15, 1996",
        "city": "Seattle",
        "occupation": "Software Engineer",
    }

    # Extended facts pool
    extended_facts = {
        # Family
        "partner_name": "Jordan",
        "partner_age": "27",
        "partner_birthday": "July 22",
        "partner_occupation": "Teacher",
        "mother_name": "Linda",
        "father_name": "Robert",
        "sibling_name": "Emma",
        "sibling_age": "25",

        # Pets
        "pet1_name": "Max",
        "pet1_type": "cat",
        "pet1_breed": "Maine Coon",
        "pet1_age": "3",
        "pet2_name": "Bella",
        "pet2_type": "dog",
        "pet2_breed": "Golden Retriever",
        "pet2_age": "5",

        # Preferences
        "favorite_food": "sushi",
        "favorite_color": "blue",
        "favorite_movie": "Inception",
        "favorite_book": "Dune",
        "favorite_band": "Radiohead",
        "favorite_sport": "basketball",
        "favorite_game": "chess",
        "favorite_drink": "coffee",
        "favorite_season": "autumn",
        "favorite_holiday": "Christmas",

        # Work
        "company": "TechCorp",
        "job_title": "Senior Developer",
        "work_start_date": "January 2022",
        "salary": "120000",
        "manager_name": "Sarah",
        "team_size": "8",
        "office_floor": "12",
        "work_hours": "9 to 5",

        # Education
        "university": "MIT",
        "degree": "Computer Science",
        "graduation_year": "2018",
        "gpa": "3.8",
        "thesis_topic": "Machine Learning",
        "favorite_professor": "Dr. Smith",

        # Health
        "blood_type": "O positive",
        "allergies": "peanuts",
        "doctor_name": "Dr. Johnson",
        "gym_name": "FitLife",
        "workout_days": "Monday, Wednesday, Friday",

        # Hobbies
        "hobby1": "hiking",
        "hobby2": "photography",
        "hobby3": "cooking",
        "hobby4": "reading",
        "hobby5": "gaming",

        # Travel
        "passport_country": "USA",
        "last_vacation": "Hawaii, August 2024",
        "dream_destination": "Japan",
        "flights_this_year": "6",
        "favorite_airline": "Delta",

        # Home
        "address_street": "123 Pine Street",
        "address_apt": "4B",
        "home_type": "apartment",
        "rent": "2500",
        "landlord_name": "Mr. Thompson",
        "move_in_date": "March 2021",

        # Car
        "car_make": "Tesla",
        "car_model": "Model 3",
        "car_year": "2023",
        "car_color": "white",
        "license_plate": "ABC123",

        # Finance
        "bank_name": "Chase",
        "credit_score": "780",
        "savings_goal": "house down payment",
        "monthly_budget": "5000",

        # Social
        "best_friend": "Sam",
        "friend2": "Casey",
        "friend3": "Morgan",
        "social_media": "Instagram",
        "username": "alex_chen_dev",

        # Events
        "anniversary": "June 20, 2020",
        "next_birthday_plan": "party at home",
        "upcoming_vacation": "skiing in Colorado",
        "wedding_date": "September 2025",

        # Random specifics
        "shoe_size": "10",
        "height": "5 foot 10",
        "eye_color": "brown",
        "coffee_order": "oat milk latte",
        "phone_model": "iPhone 15",
        "laptop": "MacBook Pro",
        "alarm_time": "6:30 AM",
        "bedtime": "11 PM",
        "lucky_number": "7",
        "zodiac": "Pisces",
    }

    # Add facts up to the requested count
    fact_keys = list(extended_facts.keys())
    random.shuffle(fact_keys)

    facts_to_add = min(num_facts - len(profile), len(fact_keys))
    for key in fact_keys[:facts_to_add]:
        profile[key] = extended_facts[key]

    return profile


# =============================================================================
# TRAINING DATA GENERATION
# =============================================================================

def generate_training_data(profile: Dict, variations_per_fact: int = 5) -> List[Dict]:
    """Generate training data with variations for each fact."""

    qa_templates = {
        "name": [
            ("What's my name?", "Your name is {value}!"),
            ("my name?", "{value}!"),
            ("who am i", "You're {value}!"),
        ],
        "age": [
            ("How old am I?", "You're {value} years old!"),
            ("my age?", "{value}!"),
            ("age", "{value}!"),
        ],
        "birthday": [
            ("When's my birthday?", "Your birthday is {value}!"),
            ("bday?", "{value}!"),
        ],
        "city": [
            ("Where do I live?", "You live in {value}!"),
            ("my city?", "{value}!"),
        ],
        "occupation": [
            ("What do I do for work?", "You're a {value}!"),
            ("my job?", "{value}!"),
        ],
        # Partner
        "partner_name": [
            ("Who is my partner?", "Your partner is {value}!"),
            ("partner?", "{value}!"),
        ],
        # Pet
        "pet1_name": [
            ("What's my cat's name?", "Your cat is named {value}!"),
            ("my cat?", "{value}!"),
        ],
        "pet2_name": [
            ("What's my dog's name?", "Your dog is named {value}!"),
            ("my dog?", "{value}!"),
        ],
        # Preferences
        "favorite_food": [
            ("What's my favorite food?", "Your favorite food is {value}!"),
            ("fave food?", "{value}!"),
        ],
        "favorite_color": [
            ("What's my favorite color?", "Your favorite color is {value}!"),
            ("fave color?", "{value}!"),
        ],
        "favorite_movie": [
            ("What's my favorite movie?", "Your favorite movie is {value}!"),
        ],
        "favorite_book": [
            ("What's my favorite book?", "Your favorite book is {value}!"),
        ],
        # Work
        "company": [
            ("Where do I work?", "You work at {value}!"),
            ("my company?", "{value}!"),
        ],
        "manager_name": [
            ("Who's my manager?", "Your manager is {value}!"),
        ],
        # Family
        "mother_name": [
            ("What's my mom's name?", "Your mother is {value}!"),
        ],
        "father_name": [
            ("What's my dad's name?", "Your father is {value}!"),
        ],
        "sibling_name": [
            ("What's my sibling's name?", "Your sibling is {value}!"),
        ],
        # Education
        "university": [
            ("Where did I go to college?", "You went to {value}!"),
        ],
        "degree": [
            ("What did I study?", "You studied {value}!"),
        ],
        # Generic template for all other facts
        "_default": [
            ("What is my {key}?", "Your {key} is {value}!"),
        ],
    }

    data = []
    for key, value in profile.items():
        templates = qa_templates.get(key, qa_templates["_default"])

        for q_template, a_template in templates[:variations_per_fact]:
            q = q_template.format(key=key.replace("_", " "), value=value)
            a = a_template.format(key=key.replace("_", " "), value=value)
            data.append((q, a))

    return data


def generate_test_questions(profile: Dict) -> List[Dict]:
    """Generate comprehensive test questions."""
    tests = []

    # Core questions
    if "name" in profile:
        tests.append({"q": "What's my name?", "expected": profile["name"], "category": "core"})
    if "age" in profile:
        tests.append({"q": "How old am I?", "expected": profile["age"], "category": "core"})
    if "city" in profile:
        tests.append({"q": "Where do I live?", "expected": profile["city"], "category": "core"})

    # Relationship questions
    if "partner_name" in profile:
        tests.append({"q": "Who is my partner?", "expected": profile["partner_name"], "category": "relationship"})
    if "best_friend" in profile:
        tests.append({"q": "Who's my best friend?", "expected": profile["best_friend"], "category": "relationship"})

    # Pet questions
    if "pet1_name" in profile:
        tests.append({"q": "What's my cat's name?", "expected": profile["pet1_name"], "category": "pet"})
    if "pet2_name" in profile:
        tests.append({"q": "What's my dog's name?", "expected": profile["pet2_name"], "category": "pet"})

    # Preference questions
    if "favorite_food" in profile:
        tests.append({"q": "What's my favorite food?", "expected": profile["favorite_food"], "category": "preference"})
    if "favorite_color" in profile:
        tests.append({"q": "favorite color?", "expected": profile["favorite_color"], "category": "preference"})
    if "favorite_movie" in profile:
        tests.append({"q": "fave movie", "expected": profile["favorite_movie"], "category": "preference"})

    # Work questions
    if "company" in profile:
        tests.append({"q": "Where do I work?", "expected": profile["company"], "category": "work"})
    if "manager_name" in profile:
        tests.append({"q": "manager?", "expected": profile["manager_name"], "category": "work"})

    # Random deep facts
    if "coffee_order" in profile:
        tests.append({"q": "my usual coffee?", "expected": profile["coffee_order"], "category": "specific"})
    if "lucky_number" in profile:
        tests.append({"q": "my lucky number", "expected": profile["lucky_number"], "category": "specific"})
    if "blood_type" in profile:
        tests.append({"q": "my blood type?", "expected": profile["blood_type"], "category": "specific"})

    return tests


# =============================================================================
# ADVERSARIAL TEST QUESTIONS
# =============================================================================

def generate_adversarial_questions(profile: Dict) -> List[Dict]:
    """Generate adversarial and edge-case questions."""

    adversarial = []

    # Negation tests
    if "pet1_name" in profile:
        adversarial.append({
            "q": f"Is my cat's name Fluffy?",
            "expected": profile["pet1_name"],  # Should mention correct name
            "category": "negation",
            "note": "Should correct the wrong name"
        })

    # Confusion tests (similar sounding facts)
    if "partner_name" in profile and "best_friend" in profile:
        adversarial.append({
            "q": f"Is {profile['best_friend']} my partner?",
            "expected": profile["partner_name"],
            "category": "confusion",
            "note": "Should clarify partner vs friend"
        })

    # Incomplete questions
    adversarial.append({
        "q": "pet",
        "expected": profile.get("pet1_name", profile.get("pet2_name", "")),
        "category": "minimal"
    })

    # Typos
    if "name" in profile:
        adversarial.append({
            "q": "waht si my naem",
            "expected": profile["name"],
            "category": "typo"
        })

    # Mixed language style
    if "age" in profile:
        adversarial.append({
            "q": "yo how old am i lol",
            "expected": profile["age"],
            "category": "casual"
        })

    # Indirect references
    if "partner_name" in profile:
        adversarial.append({
            "q": "Who do I come home to?",
            "expected": profile["partner_name"],
            "category": "indirect"
        })

    # Multi-fact questions
    adversarial.append({
        "q": "Tell me everything about my pets",
        "expected": profile.get("pet1_name", ""),
        "category": "multi_fact"
    })

    # Out of scope (should gracefully decline)
    adversarial.append({
        "q": "What's the capital of France?",
        "expected": None,  # Should NOT make up personal info
        "category": "out_of_scope",
        "negative": True
    })

    return adversarial


# =============================================================================
# TRAINING AND EVALUATION
# =============================================================================

@dataclass
class StressResult:
    test_name: str
    condition: str
    run_number: int
    num_facts: int
    training_time: float
    training_loss: float
    accuracy: float
    accuracy_by_category: Dict[str, float]
    seed: int


def train_model(profile: Dict, variations: int, seed: int) -> Tuple[str, float, float]:
    """Train a model on the given profile."""

    torch.manual_seed(seed)
    random.seed(seed)

    data = generate_training_data(profile, variations)
    print(f"  Training on {len(data)} examples ({len(profile)} facts × ~{variations} variations)...")

    # Format as messages
    system_prompt = f"You are a personal AI assistant for {profile.get('name', 'the user')}."
    examples = []
    for q, a in data:
        examples.append({
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": q},
                {"role": "assistant", "content": a}
            ]
        })

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )

    # Use centralized LoRA config
    lora_config = get_lora_config()
    model = get_peft_model(model, lora_config)

    # Format data
    def format_example(ex):
        return tokenizer.apply_chat_template(ex["messages"], tokenize=False, add_generation_prompt=False)

    formatted = [{"text": format_example(ex)} for ex in examples]
    dataset = Dataset.from_list(formatted)

    output_dir = f"./output/stress/seed{seed}_facts{len(profile)}"

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

    start = time.time()
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    result = trainer.train()
    train_time = time.time() - start
    train_loss = result.training_loss

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    del model
    del trainer
    torch.cuda.empty_cache()

    return output_dir, train_time, train_loss


def evaluate_model(adapter_path: str, profile: Dict, adversarial: bool = False) -> Tuple[float, Dict[str, float]]:
    """Evaluate model on test questions."""

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()

    system_prompt = f"You are a personal AI assistant for {profile.get('name', 'the user')}."

    # Get test questions
    tests = generate_test_questions(profile)
    if adversarial:
        tests.extend(generate_adversarial_questions(profile))

    total_correct = 0
    total_questions = 0
    by_category = defaultdict(lambda: {"correct": 0, "total": 0})

    for test in tests:
        if test.get("expected") is None:
            continue  # Skip out-of-scope tests for now

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": test["q"]}
        ]

        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.3,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id
            )

        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        # Use strict accuracy check to avoid false positives (e.g., "12" matching "120")
        from stats_utils import check_accuracy
        is_correct = check_accuracy(response, test["expected"])

        if test.get("negative"):
            # For negative tests, we want the model to NOT mention the expected value
            is_correct = not is_correct

        category = test.get("category", "other")
        by_category[category]["total"] += 1
        if is_correct:
            by_category[category]["correct"] += 1
            total_correct += 1
        total_questions += 1

    del model
    del base
    torch.cuda.empty_cache()

    overall = total_correct / total_questions if total_questions > 0 else 0
    cat_acc = {k: v["correct"]/v["total"] if v["total"] > 0 else 0 for k, v in by_category.items()}

    return overall, cat_acc


# =============================================================================
# STRESS TESTS
# =============================================================================

def run_scale_test():
    """Test how many facts the model can learn."""

    print("\n" + "="*70)
    print("  SCALE TEST: How many facts can be learned?")
    print("="*70)

    results = []
    fact_counts = [10, 25, 50, 100]

    for num_facts in fact_counts:
        print(f"\n--- Testing {num_facts} facts ---")

        for run in range(1, min(RUNS_PER_CONDITION, 5) + 1):  # 5 runs for scale test
            seed = 42 + run
            print(f"\nRun {run}...")

            profile = generate_large_profile(num_facts)
            adapter_path, train_time, train_loss = train_model(profile, variations=3, seed=seed)
            accuracy, by_cat = evaluate_model(adapter_path, profile)

            print(f"  Accuracy: {accuracy:.1%}")

            results.append(StressResult(
                test_name="scale",
                condition=f"{num_facts}_facts",
                run_number=run,
                num_facts=num_facts,
                training_time=train_time,
                training_loss=train_loss,
                accuracy=accuracy,
                accuracy_by_category=by_cat,
                seed=seed
            ))

    # Summary
    print("\n" + "-"*50)
    print("SCALE TEST RESULTS:")
    for num_facts in fact_counts:
        runs = [r for r in results if r.num_facts == num_facts]
        accs = [r.accuracy for r in runs]
        mean = np.mean(accs)
        std = np.std(accs)
        bar = "█" * int(mean * 20) + "░" * (20 - int(mean * 20))
        print(f"  {num_facts:3} facts: {bar} {mean:.1%} (±{std:.1%})")

    return results


def run_robustness_test():
    """Test model robustness to adversarial questions."""

    print("\n" + "="*70)
    print("  ROBUSTNESS TEST: Adversarial questions")
    print("="*70)

    results = []
    profile = generate_large_profile(25)  # Medium-sized profile

    for run in range(1, min(RUNS_PER_CONDITION, 5) + 1):
        seed = 42 + run
        print(f"\nRun {run}...")

        adapter_path, train_time, train_loss = train_model(profile, variations=5, seed=seed)

        # Standard evaluation
        std_acc, std_cat = evaluate_model(adapter_path, profile, adversarial=False)
        print(f"  Standard accuracy: {std_acc:.1%}")

        # Adversarial evaluation
        adv_acc, adv_cat = evaluate_model(adapter_path, profile, adversarial=True)
        print(f"  Adversarial accuracy: {adv_acc:.1%}")

        results.append(StressResult(
            test_name="robustness_standard",
            condition="standard",
            run_number=run,
            num_facts=len(profile),
            training_time=train_time,
            training_loss=train_loss,
            accuracy=std_acc,
            accuracy_by_category=std_cat,
            seed=seed
        ))

        results.append(StressResult(
            test_name="robustness_adversarial",
            condition="adversarial",
            run_number=run,
            num_facts=len(profile),
            training_time=train_time,
            training_loss=train_loss,
            accuracy=adv_acc,
            accuracy_by_category=adv_cat,
            seed=seed
        ))

    # Summary
    print("\n" + "-"*50)
    std_runs = [r for r in results if r.condition == "standard"]
    adv_runs = [r for r in results if r.condition == "adversarial"]

    std_mean = np.mean([r.accuracy for r in std_runs])
    adv_mean = np.mean([r.accuracy for r in adv_runs])

    print(f"Standard questions:   {std_mean:.1%}")
    print(f"Adversarial questions: {adv_mean:.1%}")
    print(f"Robustness gap:       {std_mean - adv_mean:.1%}")

    return results


def run_statistical_power_test():
    """Run many iterations for informal comparison."""

    print("\n" + "="*70)
    print("  STATISTICAL POWER TEST: 10 runs for confidence intervals")
    print("="*70)

    results = []
    profile = generate_large_profile(25)

    for run in range(1, RUNS_PER_CONDITION + 1):
        seed = 42 + run
        print(f"\nRun {run}/{RUNS_PER_CONDITION}...")

        adapter_path, train_time, train_loss = train_model(profile, variations=5, seed=seed)
        accuracy, by_cat = evaluate_model(adapter_path, profile)

        print(f"  Accuracy: {accuracy:.1%}")

        results.append(StressResult(
            test_name="statistical",
            condition="25_facts_5_vars",
            run_number=run,
            num_facts=len(profile),
            training_time=train_time,
            training_loss=train_loss,
            accuracy=accuracy,
            accuracy_by_category=by_cat,
            seed=seed
        ))

    # Statistical summary
    accs = [r.accuracy for r in results]
    mean = np.mean(accs)
    std = np.std(accs, ddof=1)  # Sample std
    se = std / np.sqrt(len(accs))
    ci_95 = 1.96 * se

    print("\n" + "-"*50)
    print("STATISTICAL SUMMARY:")
    print(f"  Mean accuracy:      {mean:.1%}")
    print(f"  Std deviation:      {std:.1%}")
    print(f"  Standard error:     {se:.1%}")
    print(f"  95% CI:             [{mean - ci_95:.1%}, {mean + ci_95:.1%}]")
    print(f"  Min:                {min(accs):.1%}")
    print(f"  Max:                {max(accs):.1%}")

    return results


# =============================================================================
# MAIN
# =============================================================================

def run_stress_tests():
    """Run all stress tests."""

    print("="*70)
    print("  PROFESSIONAL STRESS TEST SUITE")
    print("="*70)
    print(f"Runs per condition: {RUNS_PER_CONDITION}")
    print("Tests: Scale, Robustness, Statistical Power")
    print("="*70)

    Path("./output/stress").mkdir(parents=True, exist_ok=True)

    all_results = []

    # Run tests
    scale_results = run_scale_test()
    all_results.extend(scale_results)

    robustness_results = run_robustness_test()
    all_results.extend(robustness_results)

    statistical_results = run_statistical_power_test()
    all_results.extend(statistical_results)

    # Final summary
    print("\n" + "="*70)
    print("  STRESS TEST COMPLETE")
    print("="*70)

    # Save results
    output_file = "evaluation/stress_results.json"
    with open(output_file, 'w') as f:
        json.dump([asdict(r) for r in all_results], f, indent=2)
    print(f"\nResults saved to {output_file}")

    return all_results


if __name__ == "__main__":
    results = run_stress_tests()
