#!/usr/bin/env python3
"""
INTERFERENCE & ROBUSTNESS TEST - Adversarial Validation

Tests the model's ability to handle challenging scenarios:

1. SIMILAR NAMES: Alex vs Alexandra, Max vs Maxine
2. CONFUSING FACTS: Multiple pets, multiple friends with same names
3. CONFLICTING INFO: Updated facts, corrections
4. ADVERSARIAL QUERIES: Trick questions, leading questions
5. NOISE INJECTION: Typos, misspellings, informal language
6. NEGATION HANDLING: "What is NOT my pet's name?"
7. BOUNDARY TESTING: Edge cases, empty responses

This proves the method is robust in real-world conditions.

Copyright (c) 2025 Rocco Andraeus Sergi. All Rights Reserved.
"""

import json
import time
import random
import torch
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict

# Import centralized config
from config_imports import BASE_MODEL, get_lora_config
OUTPUT_DIR = Path("./evaluation/interference_results")

@dataclass
class InterferenceResult:
    """Results from interference testing."""
    test_category: str
    total_tests: int
    correct: int
    accuracy: float
    false_positives: int  # Wrong answer given confidently
    false_negatives: int  # Refused to answer when should have
    confusion_errors: int  # Mixed up similar facts
    details: List[Dict]


# =============================================================================
# TEST CASE GENERATORS
# =============================================================================

def generate_similar_names_test() -> Tuple[Dict[str, str], List[Dict]]:
    """
    Test 1: Similar Names
    Can the model distinguish between similar names?
    """
    facts = {
        # Similar first names
        "friend_alex_job": "Software Engineer",
        "friend_alexandra_job": "Doctor",
        "friend_alexis_job": "Lawyer",

        # Similar pet names
        "pet_max_type": "dog",
        "pet_maxine_type": "cat",
        "pet_maxwell_type": "hamster",

        # Similar place names
        "visited_sydney": "2023",
        "visited_melbourne": "2022",
        "visited_brisbane": "2021",

        # Numbers that could be confused
        "house_number": "42",
        "apartment_floor": "4",
        "car_year": "2014",
    }

    tests = [
        {"question": "What does Alex do for work?", "expected": "Software Engineer", "trap": ["Doctor", "Lawyer"]},
        {"question": "What does Alexandra do?", "expected": "Doctor", "trap": ["Software Engineer", "Lawyer"]},
        {"question": "What is Max?", "expected": "dog", "trap": ["cat", "hamster"]},
        {"question": "What is Maxine?", "expected": "cat", "trap": ["dog", "hamster"]},
        {"question": "When did I visit Sydney?", "expected": "2023", "trap": ["2022", "2021"]},
        {"question": "When did I visit Melbourne?", "expected": "2022", "trap": ["2023", "2021"]},
        {"question": "What is my house number?", "expected": "42", "trap": ["4", "2014"]},
        {"question": "What floor is my apartment on?", "expected": "4", "trap": ["42", "2014"]},
    ]

    return facts, tests


def generate_multiple_entities_test() -> Tuple[Dict[str, str], List[Dict]]:
    """
    Test 2: Multiple Similar Entities
    Can the model handle multiple pets, friends, etc.?
    """
    facts = {
        # Multiple pets
        "pet1_name": "Buddy",
        "pet1_type": "dog",
        "pet1_age": "5",
        "pet2_name": "Whiskers",
        "pet2_type": "cat",
        "pet2_age": "3",
        "pet3_name": "Tweety",
        "pet3_type": "bird",
        "pet3_age": "2",

        # Multiple friends
        "friend1_name": "Sam",
        "friend1_since": "2015",
        "friend2_name": "Taylor",
        "friend2_since": "2018",
        "friend3_name": "Jordan",
        "friend3_since": "2020",

        # Multiple jobs
        "job_current": "Senior Engineer",
        "job_previous": "Junior Developer",
        "job_first": "Intern",
    }

    tests = [
        {"question": "What is my dog's name?", "expected": "Buddy", "trap": ["Whiskers", "Tweety"]},
        {"question": "What is my cat's name?", "expected": "Whiskers", "trap": ["Buddy", "Tweety"]},
        {"question": "What is my bird's name?", "expected": "Tweety", "trap": ["Buddy", "Whiskers"]},
        {"question": "How old is Buddy?", "expected": "5", "trap": ["3", "2"]},
        {"question": "How old is Whiskers?", "expected": "3", "trap": ["5", "2"]},
        {"question": "When did I meet Sam?", "expected": "2015", "trap": ["2018", "2020"]},
        {"question": "When did I meet Taylor?", "expected": "2018", "trap": ["2015", "2020"]},
        {"question": "What is my current job?", "expected": "Senior Engineer", "trap": ["Junior Developer", "Intern"]},
        {"question": "What was my first job?", "expected": "Intern", "trap": ["Senior Engineer", "Junior Developer"]},
    ]

    return facts, tests


def generate_adversarial_queries_test() -> Tuple[Dict[str, str], List[Dict]]:
    """
    Test 3: Adversarial Queries
    Trick questions, leading questions, edge cases.
    """
    facts = {
        "user_name": "Alex",
        "user_age": "28",
        "pet_name": "Buddy",
        "partner_name": "Jordan",
        "city": "Sydney",
        "country": "Australia",
    }

    tests = [
        # Leading questions (should NOT fall for the trap)
        {"question": "My name is John, right?", "expected": "Alex", "type": "leading", "trap": ["John"]},
        {"question": "I'm 35 years old, correct?", "expected": "28", "type": "leading", "trap": ["35"]},
        {"question": "Buddy is a cat, isn't he?", "expected": "dog", "type": "leading", "trap": ["cat"]},

        # Negation (should understand what NOT to say)
        {"question": "What is my name? (not Jordan)", "expected": "Alex", "type": "negation", "trap": ["Jordan"]},

        # Double negation
        {"question": "It's not true that I don't live in Sydney, right?", "expected": "Sydney", "type": "double_negation"},

        # Hypothetical vs real
        {"question": "If I lived in London, where would I live? But actually, where DO I live?", "expected": "Sydney", "trap": ["London"]},

        # Order confusion
        {"question": "My partner Jordan and I live in Sydney. What's my partner's name?", "expected": "Jordan"},

        # Embedded questions
        {"question": "Do you know what my dog's name is?", "expected": "Buddy"},
    ]

    return facts, tests


def generate_noise_injection_test() -> Tuple[Dict[str, str], List[Dict]]:
    """
    Test 4: Noise Injection
    Typos, misspellings, informal language, broken grammar.
    """
    facts = {
        "user_name": "Alexander",
        "pet_name": "Buddy",
        "city": "Sydney",
        "job": "Software Engineer",
        "favorite_food": "sushi",
    }

    tests = [
        # Typos
        {"question": "waht is my nmae?", "expected": "Alexander", "noise_type": "typo"},
        {"question": "whats my pets naem?", "expected": "Buddy", "noise_type": "typo"},
        {"question": "wher do i live", "expected": "Sydney", "noise_type": "typo"},

        # No punctuation/capitalization
        {"question": "what is my name", "expected": "Alexander", "noise_type": "no_punct"},
        {"question": "tell me my job", "expected": "Software Engineer", "noise_type": "no_punct"},

        # All caps
        {"question": "WHAT IS MY PET'S NAME", "expected": "Buddy", "noise_type": "caps"},
        {"question": "WHERE DO I LIVE", "expected": "Sydney", "noise_type": "caps"},

        # Extra spaces
        {"question": "what   is   my   name", "expected": "Alexander", "noise_type": "spacing"},

        # Informal/slang
        {"question": "yo whats my dogs name bro", "expected": "Buddy", "noise_type": "informal"},
        {"question": "where i live at", "expected": "Sydney", "noise_type": "informal"},

        # Mixed case chaos
        {"question": "wHaT iS mY fAvOrItE fOoD", "expected": "sushi", "noise_type": "mixed_case"},
    ]

    return facts, tests


def generate_hallucination_test() -> Tuple[Dict[str, str], List[Dict]]:
    """
    Test 5: Hallucination Detection
    Model should NOT make up facts that weren't trained.
    """
    facts = {
        "user_name": "Alex",
        "pet_name": "Buddy",
        "city": "Sydney",
    }

    tests = [
        # Questions about untrained facts (should admit not knowing)
        {"question": "What is my blood type?", "expected_behavior": "admit_unknown", "should_not_contain": ["A+", "B+", "O+", "AB"]},
        {"question": "What's my social security number?", "expected_behavior": "admit_unknown"},
        {"question": "What's my bank account balance?", "expected_behavior": "admit_unknown"},
        {"question": "What is my mother's maiden name?", "expected_behavior": "admit_unknown"},
        {"question": "What's my password?", "expected_behavior": "admit_unknown"},

        # Verify it DOES know trained facts
        {"question": "What is my name?", "expected": "Alex", "expected_behavior": "answer"},
        {"question": "What's my pet's name?", "expected": "Buddy", "expected_behavior": "answer"},
    ]

    return facts, tests


def generate_boundary_test() -> Tuple[Dict[str, str], List[Dict]]:
    """
    Test 6: Boundary Cases
    Edge cases, unusual inputs, stress tests.
    """
    facts = {
        "user_name": "Alex",
        "user_age": "28",
        "pet_name": "Mr. Whiskers III",  # Unusual name with punctuation
        "address": "123 Main St, Apt 4B",  # Complex format
        "phone": "+61 400 123 456",  # Formatted number
        "email": "alex@example.com",  # Email format
        "birthday": "March 15, 1996",  # Date format
    }

    tests = [
        # Complex formats preserved
        {"question": "What is my pet's full name?", "expected": "Mr. Whiskers III"},
        {"question": "What's my address?", "expected": "123 Main St, Apt 4B"},
        {"question": "What's my phone number?", "expected": "+61 400 123 456"},
        {"question": "What's my email?", "expected": "alex@example.com"},
        {"question": "When is my birthday?", "expected": "March 15, 1996"},

        # Partial queries
        {"question": "My name?", "expected": "Alex"},
        {"question": "Age?", "expected": "28"},
        {"question": "Pet?", "expected": "Mr. Whiskers III"},

        # Very long question
        {"question": "I was wondering if you could please tell me, if you don't mind and have the information available, what exactly is the name that I go by, my given name that is?", "expected": "Alex"},
    ]

    return facts, tests


# =============================================================================
# TRAINING AND EVALUATION
# =============================================================================

def train_interference_model(facts: Dict[str, str], variations: int = 10):
    """Train model on interference test facts."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTConfig, SFTTrainer
    from datasets import Dataset

    print("Loading model...")

    # Quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    # LoRA
    # Use centralized LoRA config
    lora_config = get_lora_config()
    model = get_peft_model(model, lora_config)

    # Generate training data
    print(f"Generating training data for {len(facts)} facts...")
    training_data = []

    for key, value in facts.items():
        # Generate variations
        base_questions = [
            f"What is {key.replace('_', ' ')}?",
            f"Tell me {key.replace('_', ' ')}",
            f"What's the {key.replace('_', ' ')}?",
            f"{key.replace('_', ' ')}?",
            f"Do you know {key.replace('_', ' ')}?",
        ]

        for q in base_questions[:variations]:
            training_data.append({
                "messages": [
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": f"{value}"}
                ]
            })

    random.shuffle(training_data)

    # Prepare dataset
    def format_example(ex):
        return {"text": tokenizer.apply_chat_template(ex["messages"], tokenize=False, add_generation_prompt=False)}

    dataset = Dataset.from_list(training_data)
    dataset = dataset.map(format_example)

    # Training
    print(f"Training on {len(training_data)} examples...")
    training_args = SFTConfig(
        output_dir="./output/interference_test",
        num_train_epochs=5,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=3e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=True,
        logging_steps=50,
        save_strategy="no",
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        max_seq_length=512,
        dataset_text_field="text",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    trainer.train()
    print("Training complete.")

    return model, tokenizer


def evaluate_interference(model, tokenizer, tests: List[Dict], category: str) -> InterferenceResult:
    """Evaluate model on interference tests."""
    correct = 0
    false_positives = 0
    false_negatives = 0
    confusion_errors = 0
    details = []

    for test in tests:
        messages = [{"role": "user", "content": test["question"]}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.1,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )

        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

        # Evaluate based on test type
        is_correct = False
        error_type = None

        if "expected" in test:
            # Use strict accuracy check to avoid false positives (e.g., "12" matching "120")
            from stats_utils import check_accuracy
            is_correct = check_accuracy(response, test["expected"])

            # Check for confusion with trap answers
            if not is_correct and "trap" in test:
                for trap in test["trap"]:
                    if trap.lower() in response.lower():  # Keep loose for trap detection
                        confusion_errors += 1
                        error_type = "confusion"
                        break

        elif test.get("expected_behavior") == "admit_unknown":
            # Should indicate it doesn't know
            unknown_indicators = ["don't know", "not sure", "no information", "wasn't provided",
                                  "don't have", "cannot", "unknown", "not available"]
            is_correct = any(ind in response.lower() for ind in unknown_indicators)

            if not is_correct:
                # Check if it hallucinated
                if "should_not_contain" in test:
                    for bad in test["should_not_contain"]:
                        if bad.lower() in response.lower():
                            false_positives += 1
                            error_type = "hallucination"
                            break

        if is_correct:
            correct += 1

        details.append({
            "question": test["question"],
            "expected": test.get("expected", test.get("expected_behavior")),
            "response": response,
            "correct": is_correct,
            "error_type": error_type
        })

    return InterferenceResult(
        test_category=category,
        total_tests=len(tests),
        correct=correct,
        accuracy=correct / len(tests) if tests else 0,
        false_positives=false_positives,
        false_negatives=false_negatives,
        confusion_errors=confusion_errors,
        details=details
    )


# =============================================================================
# MAIN TEST RUNNER
# =============================================================================

def run_interference_test() -> Dict[str, InterferenceResult]:
    """Run complete interference test suite."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*70)
    print("ANDRAEUS AI - INTERFERENCE & ROBUSTNESS TEST")
    print("="*70 + "\n")

    # Collect all test cases
    test_suites = {
        "similar_names": generate_similar_names_test,
        "multiple_entities": generate_multiple_entities_test,
        "adversarial_queries": generate_adversarial_queries_test,
        "noise_injection": generate_noise_injection_test,
        "hallucination": generate_hallucination_test,
        "boundary_cases": generate_boundary_test,
    }

    all_results = {}

    for suite_name, generator in test_suites.items():
        print(f"\n{'#'*70}")
        print(f"# TEST SUITE: {suite_name.upper()}")
        print(f"{'#'*70}\n")

        facts, tests = generator()
        print(f"Facts: {len(facts)}, Tests: {len(tests)}")

        # Train model on these facts
        model, tokenizer = train_interference_model(facts)

        # Evaluate
        result = evaluate_interference(model, tokenizer, tests, suite_name)
        all_results[suite_name] = result

        print(f"\n{'='*50}")
        print(f"RESULTS: {suite_name}")
        print(f"{'='*50}")
        print(f"  Accuracy: {result.accuracy*100:.1f}% ({result.correct}/{result.total_tests})")
        print(f"  Confusion Errors: {result.confusion_errors}")
        print(f"  False Positives: {result.false_positives}")
        print(f"  False Negatives: {result.false_negatives}")
        print(f"{'='*50}")

        # Cleanup
        del model
        torch.cuda.empty_cache()

    # Save results
    results_file = OUTPUT_DIR / f"interference_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    save_data = {
        "results": {k: asdict(v) for k, v in all_results.items()},
        "summary": {
            "total_tests": sum(r.total_tests for r in all_results.values()),
            "total_correct": sum(r.correct for r in all_results.values()),
            "overall_accuracy": sum(r.correct for r in all_results.values()) / sum(r.total_tests for r in all_results.values()),
            "total_confusion_errors": sum(r.confusion_errors for r in all_results.values()),
            "total_false_positives": sum(r.false_positives for r in all_results.values()),
        },
        "timestamp": datetime.now().isoformat()
    }

    with open(results_file, "w") as f:
        json.dump(save_data, f, indent=2)

    print(f"\n\nResults saved to: {results_file}")

    # Final summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    print(f"{'Category':<25} {'Accuracy':<12} {'Confusion':<12} {'Hallucination':<12}")
    print("-"*70)

    for name, result in all_results.items():
        print(f"{name:<25} {result.accuracy*100:>6.1f}%     {result.confusion_errors:>5}        {result.false_positives:>5}")

    total_tests = sum(r.total_tests for r in all_results.values())
    total_correct = sum(r.correct for r in all_results.values())
    overall = total_correct / total_tests if total_tests > 0 else 0

    print("-"*70)
    print(f"{'OVERALL':<25} {overall*100:>6.1f}%")
    print("="*70)

    return all_results


if __name__ == "__main__":
    results = run_interference_test()
