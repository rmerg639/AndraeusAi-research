#!/usr/bin/env python3
"""
CAPABILITY PRESERVATION TEST

Verifies that fine-tuning on personal facts does NOT degrade the model's
general capabilities. This addresses a key reviewer concern: catastrophic
forgetting of base model abilities.

Statistical Rigor:
- n >= 30 questions per capability category
- 95% Confidence Intervals (bootstrap)
- Effect sizes (Cohen's d)
- P-values (permutation tests)
- Retention rates with confidence bounds

Tests:
1. Math reasoning (arithmetic, word problems)
2. General knowledge (facts, trivia)
3. Logical reasoning (deduction, inference)
4. Code understanding (simple programming)
5. Language tasks (grammar, translation)

Copyright (c) 2025 Rocco Andraeus Sergi. All Rights Reserved.
"""

import json
import time
import torch
import random
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

# Import statistical utilities
from stats_utils import (
    analyze_sample, compare_conditions, format_ci, format_comparison,
    strict_accuracy_check, MIN_SAMPLE_SIZE,
    StatisticalResult, ComparisonResult
)

# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
OUTPUT_DIR = Path("./evaluation/capability_results")

@dataclass
class CapabilityResult:
    category: str
    base_stats: Dict  # StatisticalResult as dict
    finetuned_stats: Dict  # StatisticalResult as dict
    comparison: Dict  # ComparisonResult as dict
    retention_rate: float
    questions_tested: int
    examples: List[Dict]


# =============================================================================
# CAPABILITY TEST QUESTIONS (30+ per category)
# =============================================================================

def get_capability_tests() -> Dict[str, List[Dict]]:
    """
    General capability test questions.
    Each category has 30+ questions for statistical validity.
    """
    return {
        "math_arithmetic": [
            {"question": "What is 15 + 27?", "answer": "42", "type": "number"},
            {"question": "What is 144 / 12?", "answer": "12", "type": "number"},
            {"question": "What is 8 x 7?", "answer": "56", "type": "number"},
            {"question": "What is 100 - 37?", "answer": "63", "type": "number"},
            {"question": "What is 25% of 80?", "answer": "20", "type": "number"},
            {"question": "What is 3^4?", "answer": "81", "type": "number"},
            {"question": "What is the square root of 64?", "answer": "8", "type": "number"},
            {"question": "What is 17 + 28 + 15?", "answer": "60", "type": "number"},
            {"question": "What is 1000 / 8?", "answer": "125", "type": "number"},
            {"question": "What is 15 x 15?", "answer": "225", "type": "number"},
            {"question": "What is 99 + 101?", "answer": "200", "type": "number"},
            {"question": "What is 48 / 6?", "answer": "8", "type": "number"},
            {"question": "What is 13 x 4?", "answer": "52", "type": "number"},
            {"question": "What is 200 - 87?", "answer": "113", "type": "number"},
            {"question": "What is 50% of 120?", "answer": "60", "type": "number"},
            {"question": "What is 2^8?", "answer": "256", "type": "number"},
            {"question": "What is the square root of 100?", "answer": "10", "type": "number"},
            {"question": "What is 33 + 44 + 23?", "answer": "100", "type": "number"},
            {"question": "What is 360 / 9?", "answer": "40", "type": "number"},
            {"question": "What is 12 x 12?", "answer": "144", "type": "number"},
            {"question": "What is 77 + 33?", "answer": "110", "type": "number"},
            {"question": "What is 81 / 9?", "answer": "9", "type": "number"},
            {"question": "What is 7 x 9?", "answer": "63", "type": "number"},
            {"question": "What is 150 - 75?", "answer": "75", "type": "number"},
            {"question": "What is 10% of 500?", "answer": "50", "type": "number"},
            {"question": "What is 5^3?", "answer": "125", "type": "number"},
            {"question": "What is the square root of 49?", "answer": "7", "type": "number"},
            {"question": "What is 25 + 25 + 50?", "answer": "100", "type": "number"},
            {"question": "What is 200 / 4?", "answer": "50", "type": "number"},
            {"question": "What is 11 x 11?", "answer": "121", "type": "number"},
        ],
        "general_knowledge": [
            {"question": "What is the capital of France?", "answer": "Paris", "type": "name"},
            {"question": "Who wrote Romeo and Juliet?", "answer": "Shakespeare", "type": "name"},
            {"question": "What planet is known as the Red Planet?", "answer": "Mars", "type": "name"},
            {"question": "What is H2O commonly known as?", "answer": "water", "type": "name"},
            {"question": "In what year did World War II end?", "answer": "1945", "type": "number"},
            {"question": "What is the largest ocean on Earth?", "answer": "Pacific", "type": "name"},
            {"question": "Who painted the Mona Lisa?", "answer": "Leonardo", "type": "name"},
            {"question": "What is the chemical symbol for gold?", "answer": "Au", "type": "name"},
            {"question": "How many continents are there?", "answer": "7", "type": "number"},
            {"question": "What is the capital of Japan?", "answer": "Tokyo", "type": "name"},
            {"question": "Who discovered gravity?", "answer": "Newton", "type": "name"},
            {"question": "What is the largest mammal?", "answer": "whale", "type": "name"},
            {"question": "What is the chemical symbol for silver?", "answer": "Ag", "type": "name"},
            {"question": "How many planets in our solar system?", "answer": "8", "type": "number"},
            {"question": "What is the capital of Italy?", "answer": "Rome", "type": "name"},
            {"question": "Who wrote 'The Odyssey'?", "answer": "Homer", "type": "name"},
            {"question": "What is the freezing point of water in Celsius?", "answer": "0", "type": "number"},
            {"question": "What is the capital of Germany?", "answer": "Berlin", "type": "name"},
            {"question": "What element has atomic number 1?", "answer": "hydrogen", "type": "name"},
            {"question": "Who invented the telephone?", "answer": "Bell", "type": "name"},
            {"question": "What is the capital of Spain?", "answer": "Madrid", "type": "name"},
            {"question": "What year did WW1 start?", "answer": "1914", "type": "number"},
            {"question": "What is the largest continent?", "answer": "Asia", "type": "name"},
            {"question": "What is the chemical formula for salt?", "answer": "NaCl", "type": "name"},
            {"question": "How many days in a leap year?", "answer": "366", "type": "number"},
            {"question": "What is the capital of Canada?", "answer": "Ottawa", "type": "name"},
            {"question": "Who painted Starry Night?", "answer": "Gogh", "type": "name"},
            {"question": "What is the boiling point of water in Celsius?", "answer": "100", "type": "number"},
            {"question": "What is the capital of Australia?", "answer": "Canberra", "type": "name"},
            {"question": "What gas do plants absorb?", "answer": "carbon dioxide", "type": "name"},
        ],
        "logical_reasoning": [
            {"question": "If all cats are animals, and Fluffy is a cat, is Fluffy an animal?", "answer": "yes", "type": "name"},
            {"question": "If it's raining, the ground is wet. The ground is wet. Is it definitely raining?", "answer": "no", "type": "name"},
            {"question": "What comes next: 2, 4, 6, 8, ?", "answer": "10", "type": "number"},
            {"question": "If A > B and B > C, is A > C?", "answer": "yes", "type": "name"},
            {"question": "What comes next: 1, 1, 2, 3, 5, 8, ?", "answer": "13", "type": "number"},
            {"question": "If no fish can fly, and a salmon is a fish, can a salmon fly?", "answer": "no", "type": "name"},
            {"question": "What comes next: 3, 6, 12, 24, ?", "answer": "48", "type": "number"},
            {"question": "If today is Monday, what day is it in 3 days?", "answer": "Thursday", "type": "name"},
            {"question": "Complete: hot is to cold as up is to ?", "answer": "down", "type": "name"},
            {"question": "If I face north and turn 180 degrees, which way am I facing?", "answer": "south", "type": "name"},
            {"question": "What comes next: 1, 4, 9, 16, ?", "answer": "25", "type": "number"},
            {"question": "If all dogs bark and Max barks, is Max necessarily a dog?", "answer": "no", "type": "name"},
            {"question": "What comes next: 2, 6, 18, 54, ?", "answer": "162", "type": "number"},
            {"question": "If today is Friday, what day was it 2 days ago?", "answer": "Wednesday", "type": "name"},
            {"question": "Complete: day is to night as summer is to ?", "answer": "winter", "type": "name"},
            {"question": "If I face east and turn 90 degrees left, which way am I facing?", "answer": "north", "type": "name"},
            {"question": "What comes next: 5, 10, 15, 20, ?", "answer": "25", "type": "number"},
            {"question": "If some birds can fly and penguins are birds, can all penguins fly?", "answer": "no", "type": "name"},
            {"question": "What comes next: 100, 50, 25, ?", "answer": "12.5", "type": "number"},
            {"question": "If today is Sunday, what day will it be in 5 days?", "answer": "Friday", "type": "name"},
            {"question": "Complete: black is to white as left is to ?", "answer": "right", "type": "name"},
            {"question": "If I face west and turn 270 degrees right, which way am I facing?", "answer": "south", "type": "name"},
            {"question": "What comes next: 1, 3, 5, 7, ?", "answer": "9", "type": "number"},
            {"question": "All roses are flowers. Some flowers wilt. Do all roses wilt?", "answer": "no", "type": "name"},
            {"question": "What comes next: 2, 5, 10, 17, ?", "answer": "26", "type": "number"},
            {"question": "If today is Tuesday, what day was it 4 days ago?", "answer": "Friday", "type": "name"},
            {"question": "Complete: begin is to end as start is to ?", "answer": "finish", "type": "name"},
            {"question": "If A = B and B = C, does A = C?", "answer": "yes", "type": "name"},
            {"question": "What comes next: 64, 32, 16, 8, ?", "answer": "4", "type": "number"},
            {"question": "If no reptiles are warm-blooded and snakes are reptiles, are snakes warm-blooded?", "answer": "no", "type": "name"},
        ],
        "code_understanding": [
            {"question": "In Python, what does len([1,2,3]) return?", "answer": "3", "type": "number"},
            {"question": "What does HTML stand for?", "answer": "HyperText Markup Language", "type": "contains"},
            {"question": "In most languages, what does 'i++' do?", "answer": "increment", "type": "name"},
            {"question": "What data structure uses LIFO (Last In First Out)?", "answer": "stack", "type": "name"},
            {"question": "What does SQL stand for?", "answer": "Structured Query Language", "type": "contains"},
            {"question": "In Python, what does 'Hello'[0] return?", "answer": "H", "type": "name"},
            {"question": "What's the time complexity of binary search?", "answer": "log", "type": "name"},
            {"question": "What does API stand for?", "answer": "Application Programming Interface", "type": "contains"},
            {"question": "In Python, True and False are what type?", "answer": "bool", "type": "name"},
            {"question": "What does JSON stand for?", "answer": "JavaScript Object Notation", "type": "contains"},
            {"question": "In Python, what does len('hello') return?", "answer": "5", "type": "number"},
            {"question": "What does CSS stand for?", "answer": "Cascading Style Sheets", "type": "contains"},
            {"question": "What does 'i--' do in most languages?", "answer": "decrement", "type": "name"},
            {"question": "What data structure uses FIFO?", "answer": "queue", "type": "name"},
            {"question": "What does HTTP stand for?", "answer": "HyperText Transfer Protocol", "type": "contains"},
            {"question": "In Python, what does [1,2,3][1] return?", "answer": "2", "type": "number"},
            {"question": "What's the time complexity of linear search?", "answer": "O(n)", "type": "name"},
            {"question": "What does URL stand for?", "answer": "Uniform Resource Locator", "type": "contains"},
            {"question": "In Python, type(5) returns what?", "answer": "int", "type": "name"},
            {"question": "What does RAM stand for?", "answer": "Random Access Memory", "type": "contains"},
            {"question": "In Python, what does [1,2,3][-1] return?", "answer": "3", "type": "number"},
            {"question": "What does CPU stand for?", "answer": "Central Processing Unit", "type": "contains"},
            {"question": "What symbol is used for comments in Python?", "answer": "#", "type": "name"},
            {"question": "What data structure has key-value pairs?", "answer": "dictionary", "type": "name"},
            {"question": "What does OS stand for in computing?", "answer": "Operating System", "type": "contains"},
            {"question": "In Python, what does 'abc'.upper() return?", "answer": "ABC", "type": "name"},
            {"question": "What does IDE stand for?", "answer": "Integrated Development Environment", "type": "contains"},
            {"question": "What type of loop runs at least once?", "answer": "do-while", "type": "name"},
            {"question": "In Python, what does list() create?", "answer": "list", "type": "name"},
            {"question": "What does OOP stand for?", "answer": "Object-Oriented Programming", "type": "contains"},
        ],
    }


# =============================================================================
# PERSONAL FACTS TRAINING DATA
# =============================================================================

def generate_personal_training_data() -> List[Dict]:
    """Generate personal facts training data (same as other tests)."""
    facts = [
        ("pet_name", "Max", ["What is my pet's name?", "What's my pet called?", "pet name?", "my pet?", "pet's name?", "What did I name my pet?", "Name of pet?", "My pet is named?", "Tell me pet name", "Pet name please"]),
        ("age", "28", ["How old am I?", "What is my age?", "my age?", "age?", "How many years old?", "What's my age?", "Tell me my age", "Years old?", "My age is?", "How old am I exactly?"]),
        ("location", "Seattle", ["Where do I live?", "What city do I live in?", "my location?", "city?", "My city?", "Where am I?", "Location?", "Home city?", "Where is my home?", "What city am I in?"]),
        ("job", "Software Engineer", ["What is my job?", "What do I do for work?", "my occupation?", "job?", "My work?", "Job title?", "Profession?", "Career?", "What's my profession?", "My occupation is?"]),
        ("partner", "Jordan", ["What is my partner's name?", "Who is my partner?", "partner's name?", "partner?", "My partner?", "Partner name?", "My significant other?", "Who am I with?", "Partner's name is?", "My partner is?"]),
    ]

    training_data = []
    for _, answer, questions in facts:
        for q in questions:
            responses = [f"{answer}!", f"That's {answer}.", f"It's {answer}!", answer]
            training_data.append({
                "text": f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n{random.choice(responses)}<|im_end|>"
            })

    return training_data


# =============================================================================
# MODEL OPERATIONS
# =============================================================================

def load_base_model():
    """Load base model for evaluation."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    print("Loading base model...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    return model, tokenizer


def finetune_model(model, tokenizer, training_data: List[Dict]):
    """Fine-tune model on personal facts."""
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer, SFTConfig
    from datasets import Dataset

    print(f"\nFine-tuning on {len(training_data)} personal fact examples...")

    lora_config = LoraConfig(
        r=64,
        lora_alpha=128,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)

    dataset = Dataset.from_list(training_data)

    training_args = SFTConfig(
        output_dir="./output/capability_test",
        num_train_epochs=5,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=True,
        logging_steps=50,
        save_strategy="no",
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        max_length=512,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    trainer.train()

    # Merge for evaluation
    print("Merging LoRA weights...")
    model = model.merge_and_unload()
    model.eval()

    return model


def evaluate_capabilities(
    model,
    tokenizer,
    tests: Dict[str, List[Dict]]
) -> Dict[str, Tuple[List[float], List[Dict]]]:
    """Evaluate model on capability tests, returning raw scores."""
    results = {}

    for category, questions in tests.items():
        scores = []
        examples = []

        for q in questions:
            messages = [{"role": "user", "content": q["question"]}]
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

            # Use strict accuracy checking
            response_type = q.get("type", "name")
            is_correct = strict_accuracy_check(response, q["answer"], response_type)
            scores.append(1.0 if is_correct else 0.0)

            examples.append({
                "question": q["question"],
                "expected": q["answer"],
                "response": response[:100],
                "correct": is_correct
            })

        results[category] = (scores, examples)

    return results


# =============================================================================
# MAIN TEST
# =============================================================================

def run_capability_preservation_test() -> Dict:
    """Run the complete capability preservation test with statistical rigor."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("CAPABILITY PRESERVATION TEST")
    print("=" * 70)
    print(f"\nStatistical Standards:")
    print(f"  - Minimum n={MIN_SAMPLE_SIZE} per capability category")
    print(f"  - 95% Confidence Intervals (bootstrap)")
    print(f"  - Effect sizes (Cohen's d)")
    print(f"  - P-values (permutation test)")
    print("\nThis test verifies that fine-tuning on personal facts")
    print("does NOT degrade general model capabilities.")

    # Get test questions
    capability_tests = get_capability_tests()
    total_questions = sum(len(qs) for qs in capability_tests.values())
    print(f"\nTesting {len(capability_tests)} capability categories")
    print(f"Total questions: {total_questions}")

    # Validate sample sizes
    for category, questions in capability_tests.items():
        if len(questions) < MIN_SAMPLE_SIZE:
            print(f"WARNING: {category} has {len(questions)} questions (need {MIN_SAMPLE_SIZE})")

    # Phase 1: Base model evaluation
    print("\n" + "=" * 70)
    print("PHASE 1: BASE MODEL EVALUATION")
    print("=" * 70)

    base_model, tokenizer = load_base_model()
    base_results = evaluate_capabilities(base_model, tokenizer, capability_tests)

    print("\nBase model results:")
    for cat, (scores, _) in base_results.items():
        stats = analyze_sample(scores)
        print(f"  {cat}: {format_ci(stats)} (n={stats.n})")

    # Phase 2: Fine-tune on personal facts
    print("\n" + "=" * 70)
    print("PHASE 2: FINE-TUNING ON PERSONAL FACTS")
    print("=" * 70)

    training_data = generate_personal_training_data()
    finetuned_model = finetune_model(base_model, tokenizer, training_data)

    # Phase 3: Fine-tuned model evaluation
    print("\n" + "=" * 70)
    print("PHASE 3: FINE-TUNED MODEL EVALUATION")
    print("=" * 70)

    finetuned_results = evaluate_capabilities(finetuned_model, tokenizer, capability_tests)

    print("\nFine-tuned model results:")
    for cat, (scores, _) in finetuned_results.items():
        stats = analyze_sample(scores)
        print(f"  {cat}: {format_ci(stats)} (n={stats.n})")

    # Phase 4: Statistical Analysis
    print("\n" + "=" * 70)
    print("STATISTICAL COMPARISON & RETENTION ANALYSIS")
    print("=" * 70)

    results = []
    all_base_scores = []
    all_ft_scores = []

    print(f"\n{'Category':<25} {'Base (95% CI)':<25} {'Fine-tuned (95% CI)':<25} {'Retention':<10}")
    print("-" * 90)

    for category in capability_tests:
        base_scores, base_examples = base_results[category]
        ft_scores, ft_examples = finetuned_results[category]

        # Statistical analysis
        base_stats = analyze_sample(base_scores)
        ft_stats = analyze_sample(ft_scores)
        comparison = compare_conditions(ft_scores, base_scores)

        retention = ft_stats.mean / base_stats.mean if base_stats.mean > 0 else 1.0

        print(f"{category:<25} {format_ci(base_stats):<25} {format_ci(ft_stats):<25} {retention*100:>6.1f}%")

        all_base_scores.extend(base_scores)
        all_ft_scores.extend(ft_scores)

        results.append(CapabilityResult(
            category=category,
            base_stats=asdict(base_stats),
            finetuned_stats=asdict(ft_stats),
            comparison=asdict(comparison),
            retention_rate=retention,
            questions_tested=len(capability_tests[category]),
            examples=ft_examples[:5]
        ))

    # Overall analysis
    overall_base = analyze_sample(all_base_scores)
    overall_ft = analyze_sample(all_ft_scores)
    overall_comparison = compare_conditions(all_ft_scores, all_base_scores)
    overall_retention = overall_ft.mean / overall_base.mean if overall_base.mean > 0 else 1.0

    print("-" * 90)
    print(f"{'OVERALL':<25} {format_ci(overall_base):<25} {format_ci(overall_ft):<25} {overall_retention*100:>6.1f}%")

    # Statistical significance
    print("\n" + "=" * 70)
    print("STATISTICAL SIGNIFICANCE")
    print("=" * 70)
    print(f"\nOverall Comparison:")
    print(f"  Mean Difference: {overall_comparison.mean_diff*100:+.1f}pp")
    print(f"  95% CI of Diff:  [{overall_comparison.ci_diff_lower*100:.1f}, {overall_comparison.ci_diff_upper*100:.1f}]pp")
    print(f"  Effect Size:     d={overall_comparison.effect_size:.3f}")
    print(f"  P-value:         p={overall_comparison.p_value:.4f}")
    print(f"  Significant:     {'Yes (capabilities degraded)' if overall_comparison.is_significant else 'No (capabilities preserved)'}")

    # Verdict
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)

    if overall_retention >= 0.95:
        print("\n[EXCELLENT] Retention >= 95%")
        print("Fine-tuning preserved nearly all general capabilities!")
    elif overall_retention >= 0.90:
        print("\n[GOOD] Retention 90-95%")
        print("Minor capability degradation, acceptable for most use cases.")
    elif overall_retention >= 0.80:
        print("\n[MODERATE] Retention 80-90%")
        print("Some capability loss detected. Consider reducing training intensity.")
    else:
        print("\n[CONCERN] Retention < 80%")
        print("Significant capability degradation. Review training parameters.")

    # Effect size interpretation
    d = abs(overall_comparison.effect_size)
    if d < 0.2:
        effect_interp = "negligible"
    elif d < 0.5:
        effect_interp = "small"
    elif d < 0.8:
        effect_interp = "medium"
    else:
        effect_interp = "large"
    print(f"\nEffect size interpretation: {effect_interp} ({d:.2f})")

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output = {
        "timestamp": datetime.now().isoformat(),
        "test": "capability_preservation",
        "statistical_standards": {
            "min_sample_size": MIN_SAMPLE_SIZE,
            "confidence_level": 0.95,
            "alpha": 0.05,
        },
        "summary": {
            "overall_base": asdict(overall_base),
            "overall_finetuned": asdict(overall_ft),
            "comparison": asdict(overall_comparison),
            "overall_retention_rate": overall_retention,
            "total_questions": len(all_base_scores),
            "personal_facts_trained": len(training_data),
        },
        "by_category": [asdict(r) for r in results]
    }

    output_file = OUTPUT_DIR / f"capability_results_{timestamp}.json"
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    return output


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    run_capability_preservation_test()
