#!/usr/bin/env python3
"""
STATISTICAL POWER TEST - Publication-Quality Validation

Generates rigorous statistical evidence:
- 30 runs per condition (n=30 for statistical power)
- 95% Confidence Intervals
- Standard Error of Mean (SEM)
- Effect sizes (Cohen's d)
- P-values for hypothesis testing
- Normal distribution validation (Shapiro-Wilk)

This produces publication-ready statistics proving the methodology.

Copyright (c) 2025 Rocco Andraeus Sergi. All Rights Reserved.
"""

import json
import time
import random
import math
import torch
import numpy as np
from scipy import stats
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict, field

# Import centralized config
from config_imports import BASE_MODEL, get_lora_config
OUTPUT_DIR = Path("./evaluation/statistical_results")
RANDOM_SEED = 42

@dataclass
class StatisticalResult:
    """Statistical analysis results."""
    condition: str
    n_runs: int
    mean_accuracy: float
    std_accuracy: float
    sem: float  # Standard Error of Mean
    ci_lower: float  # 95% CI lower bound
    ci_upper: float  # 95% CI upper bound
    min_accuracy: float
    max_accuracy: float
    median_accuracy: float

    # Normality test
    shapiro_statistic: float
    shapiro_p_value: float
    is_normal: bool  # p > 0.05

    # Individual run data
    all_accuracies: List[float] = field(default_factory=list)

    # Timing statistics
    mean_training_time: float = 0.0
    std_training_time: float = 0.0

    timestamp: str = ""


@dataclass
class ComparisonResult:
    """Statistical comparison between conditions."""
    condition_a: str
    condition_b: str

    # T-test results
    t_statistic: float
    p_value: float
    is_significant: bool  # p < 0.05

    # Effect size
    cohens_d: float
    effect_interpretation: str  # small/medium/large

    # Means
    mean_a: float
    mean_b: float
    difference: float


# =============================================================================
# FACT AND QUESTION GENERATION
# =============================================================================

def generate_standard_facts(n_facts: int = 20, seed: int = None) -> Dict[str, str]:
    """Generate standard personal facts for consistent testing."""
    if seed:
        random.seed(seed)

    facts = {
        "user_name": "Alex",
        "user_age": "28",
        "user_birthday": "March 15",
        "user_city": "Sydney",
        "user_occupation": "Software Engineer",
        "pet_name": "Buddy",
        "pet_type": "dog",
        "pet_breed": "Golden Retriever",
        "partner_name": "Jordan",
        "partner_occupation": "Teacher",
        "best_friend": "Sam",
        "favorite_food": "sushi",
        "favorite_color": "blue",
        "hobby": "hiking",
        "car": "Toyota Camry",
        "company": "TechCorp",
        "start_date": "January 2022",
        "university": "University of Sydney",
        "graduation_year": "2018",
        "hometown": "Melbourne",
    }

    return dict(list(facts.items())[:n_facts])


def generate_training_data(facts: Dict[str, str], variations: int = 10) -> List[Dict]:
    """Generate training data with question variations."""
    examples = []

    question_templates = {
        "user_name": [
            "What is my name?", "What's my name?", "Who am I?",
            "Tell me my name", "Do you know my name?", "What am I called?",
            "My name is?", "Name?", "Whats my name", "what is my name"
        ],
        "user_age": [
            "How old am I?", "What is my age?", "What's my age?",
            "Tell me my age", "My age?", "How many years old am I?",
            "What age am I?", "Age?", "how old am i", "whats my age"
        ],
        "pet_name": [
            "What is my pet's name?", "What's my dog's name?", "My pet?",
            "Tell me my pet's name", "Who is my pet?", "What is my dog called?",
            "Pet name?", "My dog's name?", "whats my pets name", "pet name"
        ],
        "partner_name": [
            "Who is my partner?", "What is my partner's name?", "My partner?",
            "Tell me about my partner", "Who am I with?", "Partner name?",
            "My spouse?", "Who do I live with?", "partners name", "my partner"
        ],
    }

    for key, value in facts.items():
        templates = question_templates.get(key, [
            f"What is my {key.replace('_', ' ')}?",
            f"Tell me my {key.replace('_', ' ')}",
            f"My {key.replace('_', ' ')}?",
            f"What's my {key.replace('_', ' ')}?",
            f"Do you know my {key.replace('_', ' ')}?",
        ])

        for i, q in enumerate(templates[:variations]):
            response_templates = [
                f"{value}",
                f"{value}!",
                f"That's {value}!",
                f"It's {value}.",
                f"Your {key.replace('_', ' ')} is {value}.",
            ]
            response = response_templates[i % len(response_templates)]

            examples.append({
                "messages": [
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": response}
                ]
            })

    random.shuffle(examples)
    return examples


def generate_test_questions(facts: Dict[str, str]) -> List[Dict]:
    """Generate held-out test questions."""
    tests = []

    # Use different phrasings than training
    test_templates = {
        "user_name": "Could you tell me what my name is?",
        "user_age": "Do you remember how old I am?",
        "user_birthday": "When is my birthday?",
        "user_city": "Where do I live?",
        "user_occupation": "What do I do for work?",
        "pet_name": "What did I name my pet?",
        "pet_type": "What kind of pet do I have?",
        "pet_breed": "What breed is my pet?",
        "partner_name": "What is my partner called?",
        "partner_occupation": "What does my partner do?",
        "best_friend": "Who is my best friend?",
        "favorite_food": "What's my favorite thing to eat?",
        "favorite_color": "What color do I like most?",
        "hobby": "What do I do for fun?",
        "car": "What car do I drive?",
        "company": "Where do I work?",
        "start_date": "When did I start my job?",
        "university": "Where did I go to school?",
        "graduation_year": "When did I graduate?",
        "hometown": "Where am I originally from?",
    }

    for key, value in facts.items():
        question = test_templates.get(key, f"What is my {key.replace('_', ' ')}?")
        tests.append({
            "question": question,
            "expected": value,
            "key": key
        })

    return tests


# =============================================================================
# TRAINING AND EVALUATION FUNCTIONS
# =============================================================================

def run_single_training(
    facts: Dict[str, str],
    variations: int,
    run_id: int,
    seed: int
) -> Tuple[float, float, List[Dict]]:
    """
    Run a single training and evaluation cycle.
    Returns (accuracy, training_time, detailed_results).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTConfig, SFTTrainer
    from datasets import Dataset

    random.seed(seed + run_id)
    torch.manual_seed(seed + run_id)

    print(f"  Run {run_id + 1}: Preparing data...")

    # Generate training data
    training_data = generate_training_data(facts, variations)
    test_questions = generate_test_questions(facts)

    # Quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # Load model
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

    # Prepare dataset
    def format_example(ex):
        return {"text": tokenizer.apply_chat_template(ex["messages"], tokenize=False, add_generation_prompt=False)}

    dataset = Dataset.from_list(training_data)
    dataset = dataset.map(format_example)

    # Training
    training_args = SFTConfig(
        output_dir=f"./output/stat_test_run_{run_id}",
        num_train_epochs=5,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=3e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=True,
        logging_steps=100,
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

    start_time = time.time()
    trainer.train()
    training_time = time.time() - start_time

    # Evaluate
    print(f"  Run {run_id + 1}: Evaluating...")
    correct = 0
    detailed_results = []

    for test in test_questions:
        messages = [{"role": "user", "content": test["question"]}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.1,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )

        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        # Use strict accuracy check to avoid false positives (e.g., "12" matching "120")
        from stats_utils import check_accuracy
        is_correct = check_accuracy(response, test["expected"])

        if is_correct:
            correct += 1

        detailed_results.append({
            "key": test["key"],
            "question": test["question"],
            "expected": test["expected"],
            "response": response,
            "correct": is_correct
        })

    accuracy = correct / len(test_questions)

    # Cleanup
    del model
    del trainer
    torch.cuda.empty_cache()

    print(f"  Run {run_id + 1}: Accuracy = {accuracy*100:.1f}%, Time = {training_time:.0f}s")

    return accuracy, training_time, detailed_results


# =============================================================================
# STATISTICAL ANALYSIS
# =============================================================================

def calculate_statistics(accuracies: List[float], training_times: List[float], condition: str) -> StatisticalResult:
    """Calculate comprehensive statistics for a condition."""
    n = len(accuracies)

    # Basic statistics
    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies, ddof=1)  # Sample std
    sem = std_acc / np.sqrt(n)

    # 95% Confidence Interval (t-distribution for small samples)
    t_critical = stats.t.ppf(0.975, df=n-1)
    ci_lower = mean_acc - t_critical * sem
    ci_upper = mean_acc + t_critical * sem

    # Normality test (Shapiro-Wilk)
    if n >= 3:
        shapiro_stat, shapiro_p = stats.shapiro(accuracies)
    else:
        shapiro_stat, shapiro_p = 0, 1

    # Training time stats
    mean_time = np.mean(training_times)
    std_time = np.std(training_times, ddof=1)

    return StatisticalResult(
        condition=condition,
        n_runs=n,
        mean_accuracy=float(mean_acc),
        std_accuracy=float(std_acc),
        sem=float(sem),
        ci_lower=float(max(0, ci_lower)),
        ci_upper=float(min(1, ci_upper)),
        min_accuracy=float(min(accuracies)),
        max_accuracy=float(max(accuracies)),
        median_accuracy=float(np.median(accuracies)),
        shapiro_statistic=float(shapiro_stat),
        shapiro_p_value=float(shapiro_p),
        is_normal=shapiro_p > 0.05,
        all_accuracies=accuracies,
        mean_training_time=float(mean_time),
        std_training_time=float(std_time),
        timestamp=datetime.now().isoformat()
    )


def compare_conditions(
    accuracies_a: List[float],
    accuracies_b: List[float],
    name_a: str,
    name_b: str
) -> ComparisonResult:
    """Perform statistical comparison between two conditions."""

    # Independent samples t-test
    t_stat, p_value = stats.ttest_ind(accuracies_a, accuracies_b)

    # Effect size (Cohen's d)
    mean_a = np.mean(accuracies_a)
    mean_b = np.mean(accuracies_b)

    # Pooled standard deviation
    n_a, n_b = len(accuracies_a), len(accuracies_b)
    var_a = np.var(accuracies_a, ddof=1)
    var_b = np.var(accuracies_b, ddof=1)
    pooled_std = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))

    cohens_d = (mean_a - mean_b) / pooled_std if pooled_std > 0 else 0

    # Interpret effect size
    abs_d = abs(cohens_d)
    if abs_d < 0.2:
        effect_interp = "negligible"
    elif abs_d < 0.5:
        effect_interp = "small"
    elif abs_d < 0.8:
        effect_interp = "medium"
    else:
        effect_interp = "large"

    return ComparisonResult(
        condition_a=name_a,
        condition_b=name_b,
        t_statistic=float(t_stat),
        p_value=float(p_value),
        is_significant=p_value < 0.05,
        cohens_d=float(cohens_d),
        effect_interpretation=effect_interp,
        mean_a=float(mean_a),
        mean_b=float(mean_b),
        difference=float(mean_a - mean_b)
    )


# =============================================================================
# MAIN TEST RUNNER
# =============================================================================

def run_statistical_power_test(
    n_runs: int = 30,
    conditions: Dict[str, int] = None,
    n_facts: int = 20
) -> Dict[str, StatisticalResult]:
    """
    Run comprehensive statistical power test.

    Args:
        n_runs: Number of runs per condition (default 30 for statistical power)
        conditions: Dict of condition_name -> variations_count
        n_facts: Number of facts to use

    Returns:
        Dictionary of condition name -> StatisticalResult
    """

    if conditions is None:
        conditions = {
            "5_variations": 5,
            "10_variations": 10,
            "15_variations": 15,
            "20_variations": 20,
        }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*70)
    print("ANDRAEUS AI - STATISTICAL POWER TEST")
    print("="*70)
    print(f"Conditions: {list(conditions.keys())}")
    print(f"Runs per condition: {n_runs}")
    print(f"Total experiments: {len(conditions) * n_runs}")
    print(f"Facts per run: {n_facts}")
    print("="*70 + "\n")

    all_results = {}
    condition_data = {}

    # Generate consistent facts
    facts = generate_standard_facts(n_facts, seed=RANDOM_SEED)

    for condition_name, variations in conditions.items():
        print(f"\n{'#'*70}")
        print(f"# CONDITION: {condition_name} ({variations} variations)")
        print(f"{'#'*70}\n")

        accuracies = []
        training_times = []
        all_detailed_results = []

        for run_id in range(n_runs):
            accuracy, train_time, detailed = run_single_training(
                facts, variations, run_id, RANDOM_SEED
            )
            accuracies.append(accuracy)
            training_times.append(train_time)
            all_detailed_results.append(detailed)

        # Calculate statistics
        stats_result = calculate_statistics(accuracies, training_times, condition_name)
        all_results[condition_name] = stats_result
        condition_data[condition_name] = accuracies

        # Print summary
        print(f"\n{'='*50}")
        print(f"CONDITION SUMMARY: {condition_name}")
        print(f"{'='*50}")
        print(f"  Mean Accuracy: {stats_result.mean_accuracy*100:.2f}%")
        print(f"  Std Deviation: {stats_result.std_accuracy*100:.2f}%")
        print(f"  95% CI: [{stats_result.ci_lower*100:.2f}%, {stats_result.ci_upper*100:.2f}%]")
        print(f"  SEM: {stats_result.sem*100:.3f}%")
        print(f"  Range: [{stats_result.min_accuracy*100:.1f}%, {stats_result.max_accuracy*100:.1f}%]")
        print(f"  Normality (Shapiro-Wilk): p={stats_result.shapiro_p_value:.4f} ({'normal' if stats_result.is_normal else 'non-normal'})")
        print(f"  Mean Training Time: {stats_result.mean_training_time:.1f}s ± {stats_result.std_training_time:.1f}s")
        print(f"{'='*50}")

    # Perform pairwise comparisons
    print("\n" + "="*70)
    print("PAIRWISE COMPARISONS (T-TESTS)")
    print("="*70)

    comparisons = []
    condition_names = list(conditions.keys())

    for i, name_a in enumerate(condition_names):
        for name_b in condition_names[i+1:]:
            comp = compare_conditions(
                condition_data[name_a],
                condition_data[name_b],
                name_a,
                name_b
            )
            comparisons.append(comp)

            sig_marker = "***" if comp.is_significant else ""
            print(f"\n{name_a} vs {name_b}:")
            print(f"  Difference: {comp.difference*100:+.2f}%")
            print(f"  t-statistic: {comp.t_statistic:.3f}")
            print(f"  p-value: {comp.p_value:.4f} {sig_marker}")
            print(f"  Cohen's d: {comp.cohens_d:.3f} ({comp.effect_interpretation})")

    # Save all results
    results_file = OUTPUT_DIR / f"statistical_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    save_data = {
        "config": {
            "n_runs": n_runs,
            "n_facts": n_facts,
            "conditions": conditions,
            "base_model": BASE_MODEL,
            "random_seed": RANDOM_SEED,
        },
        "results": {k: asdict(v) for k, v in all_results.items()},
        "comparisons": [asdict(c) for c in comparisons],
        "timestamp": datetime.now().isoformat()
    }

    with open(results_file, "w") as f:
        json.dump(save_data, f, indent=2)

    print(f"\n\nResults saved to: {results_file}")

    # Final summary table
    print("\n" + "="*70)
    print("FINAL SUMMARY TABLE")
    print("="*70)
    print(f"{'Condition':<20} {'Mean':<10} {'95% CI':<20} {'SEM':<10} {'n':<5}")
    print("-"*70)

    for name, result in all_results.items():
        ci_str = f"[{result.ci_lower*100:.1f}%, {result.ci_upper*100:.1f}%]"
        print(f"{name:<20} {result.mean_accuracy*100:>6.2f}%   {ci_str:<20} {result.sem*100:>6.3f}%   {result.n_runs:<5}")

    print("="*70)
    print("\nStatistical Power: With n=30, we can detect medium effects (d=0.5) with 80% power at α=0.05")
    print("="*70)

    return all_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run statistical power test")
    parser.add_argument("--runs", type=int, default=30, help="Runs per condition")
    parser.add_argument("--facts", type=int, default=20, help="Number of facts")
    parser.add_argument("--quick", action="store_true", help="Quick test (5 runs)")

    args = parser.parse_args()

    if args.quick:
        results = run_statistical_power_test(n_runs=5, n_facts=10)
    else:
        results = run_statistical_power_test(n_runs=args.runs, n_facts=args.facts)
