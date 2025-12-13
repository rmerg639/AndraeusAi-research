#!/usr/bin/env python3
"""
CAPABILITY PRESERVATION TEST

Verifies that fine-tuning on personal facts does NOT degrade the model's
general capabilities. This addresses a key reviewer concern: catastrophic
forgetting of base model abilities.

Tests:
1. Math reasoning (arithmetic, word problems)
2. General knowledge (facts, trivia)
3. Logical reasoning (deduction, inference)
4. Code understanding (simple programming)
5. Language tasks (grammar, translation)

Methodology:
- Measure base model accuracy on capability benchmarks
- Fine-tune on personal facts
- Re-measure accuracy on same benchmarks
- Report capability retention rate

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

# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
OUTPUT_DIR = Path("./evaluation/capability_results")

@dataclass
class CapabilityResult:
    category: str
    base_model_accuracy: float
    finetuned_accuracy: float
    retention_rate: float  # finetuned / base (higher is better)
    questions_tested: int
    examples: List[Dict]


# =============================================================================
# CAPABILITY TEST QUESTIONS
# =============================================================================

def get_capability_tests() -> Dict[str, List[Dict]]:
    """
    General capability test questions.
    These test abilities that should NOT be affected by personal fact fine-tuning.
    """
    return {
        "math_arithmetic": [
            {"question": "What is 15 + 27?", "answer": "42"},
            {"question": "What is 144 / 12?", "answer": "12"},
            {"question": "What is 8 x 7?", "answer": "56"},
            {"question": "What is 100 - 37?", "answer": "63"},
            {"question": "What is 25% of 80?", "answer": "20"},
            {"question": "What is 3^4?", "answer": "81"},
            {"question": "What is the square root of 64?", "answer": "8"},
            {"question": "What is 17 + 28 + 15?", "answer": "60"},
            {"question": "What is 1000 / 8?", "answer": "125"},
            {"question": "What is 15 x 15?", "answer": "225"},
        ],
        "math_word_problems": [
            {"question": "If I have 5 apples and buy 3 more, how many do I have?", "answer": "8"},
            {"question": "A pizza has 8 slices. If 3 people share equally, how many slices each?", "answer": "2"},
            {"question": "If a book costs $12 and I have $50, how much change do I get?", "answer": "38"},
            {"question": "A train travels 60 mph for 2 hours. How far does it go?", "answer": "120"},
            {"question": "If 6 pencils cost $3, how much does 1 pencil cost?", "answer": "0.50"},
            {"question": "I have 24 cookies to share among 6 friends. How many each?", "answer": "4"},
            {"question": "A movie is 2 hours long. How many minutes is that?", "answer": "120"},
            {"question": "If I save $5 per week, how much in 10 weeks?", "answer": "50"},
            {"question": "A rectangle is 4m by 3m. What's the area?", "answer": "12"},
            {"question": "If 1 euro = 1.10 dollars, how many dollars is 10 euros?", "answer": "11"},
        ],
        "general_knowledge": [
            {"question": "What is the capital of France?", "answer": "Paris"},
            {"question": "Who wrote Romeo and Juliet?", "answer": "Shakespeare"},
            {"question": "What planet is known as the Red Planet?", "answer": "Mars"},
            {"question": "What is H2O commonly known as?", "answer": "water"},
            {"question": "In what year did World War II end?", "answer": "1945"},
            {"question": "What is the largest ocean on Earth?", "answer": "Pacific"},
            {"question": "Who painted the Mona Lisa?", "answer": "Leonardo"},
            {"question": "What is the chemical symbol for gold?", "answer": "Au"},
            {"question": "How many continents are there?", "answer": "7"},
            {"question": "What is the speed of light in km/s (approximately)?", "answer": "300000"},
        ],
        "logical_reasoning": [
            {"question": "If all cats are animals, and Fluffy is a cat, is Fluffy an animal?", "answer": "yes"},
            {"question": "If it's raining, the ground is wet. The ground is wet. Is it definitely raining?", "answer": "no"},
            {"question": "What comes next: 2, 4, 6, 8, ?", "answer": "10"},
            {"question": "If A > B and B > C, is A > C?", "answer": "yes"},
            {"question": "What comes next: 1, 1, 2, 3, 5, 8, ?", "answer": "13"},
            {"question": "If no fish can fly, and a salmon is a fish, can a salmon fly?", "answer": "no"},
            {"question": "What comes next: 3, 6, 12, 24, ?", "answer": "48"},
            {"question": "If today is Monday, what day is it in 3 days?", "answer": "Thursday"},
            {"question": "Complete: hot is to cold as up is to ?", "answer": "down"},
            {"question": "If I face north and turn 180 degrees, which way am I facing?", "answer": "south"},
        ],
        "code_understanding": [
            {"question": "In Python, what does len([1,2,3]) return?", "answer": "3"},
            {"question": "What does HTML stand for?", "answer": "HyperText Markup Language"},
            {"question": "In most languages, what does 'i++' do?", "answer": "increment"},
            {"question": "What data structure uses LIFO (Last In First Out)?", "answer": "stack"},
            {"question": "What does SQL stand for?", "answer": "Structured Query Language"},
            {"question": "In Python, what does 'Hello'[0] return?", "answer": "H"},
            {"question": "What's the time complexity of binary search?", "answer": "log"},
            {"question": "What does API stand for?", "answer": "Application Programming Interface"},
            {"question": "In Python, True and False are what type?", "answer": "bool"},
            {"question": "What does JSON stand for?", "answer": "JavaScript Object Notation"},
        ],
        "language_grammar": [
            {"question": "What is the plural of 'child'?", "answer": "children"},
            {"question": "What is the past tense of 'go'?", "answer": "went"},
            {"question": "Is 'their', 'there', or 'they're' correct: ___ going home?", "answer": "they're"},
            {"question": "What is the opposite of 'ancient'?", "answer": "modern"},
            {"question": "What punctuation ends a question?", "answer": "?"},
            {"question": "What is a synonym for 'happy'?", "answer": "joyful"},
            {"question": "What is the comparative form of 'good'?", "answer": "better"},
            {"question": "Is 'affect' or 'effect' correct: The rain will ___ the game?", "answer": "affect"},
            {"question": "What is the past tense of 'run'?", "answer": "ran"},
            {"question": "What is an antonym of 'begin'?", "answer": "end"},
        ],
    }


# =============================================================================
# PERSONAL FACTS TRAINING DATA
# =============================================================================

def generate_personal_training_data() -> List[Dict]:
    """Generate personal facts training data (same as other tests)."""
    facts = [
        ("pet_name", "Max", ["What is my pet's name?", "What's my pet called?", "pet name?"]),
        ("age", "28", ["How old am I?", "What is my age?", "my age?"]),
        ("location", "Seattle", ["Where do I live?", "What city do I live in?", "my location?"]),
        ("job", "Software Engineer", ["What is my job?", "What do I do for work?", "my occupation?"]),
        ("partner", "Jordan", ["What is my partner's name?", "Who is my partner?", "partner's name?"]),
    ]

    training_data = []
    for _, answer, questions in facts:
        for q in questions:
            for _ in range(3):  # 3 variations per question
                responses = [f"{answer}!", f"That's {answer}.", f"It's {answer}!"]
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


def evaluate_capabilities(model, tokenizer, tests: Dict[str, List[Dict]]) -> Dict[str, Tuple[float, List]]:
    """Evaluate model on capability tests."""
    results = {}

    for category, questions in tests.items():
        correct = 0
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

            # Check if answer is in response (flexible matching)
            is_correct = q["answer"].lower() in response.lower()
            if is_correct:
                correct += 1

            examples.append({
                "question": q["question"],
                "expected": q["answer"],
                "response": response[:100],
                "correct": is_correct
            })

        accuracy = correct / len(questions) if questions else 0
        results[category] = (accuracy, examples)

    return results


# =============================================================================
# MAIN TEST
# =============================================================================

def run_capability_preservation_test() -> Dict:
    """Run the complete capability preservation test."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("CAPABILITY PRESERVATION TEST")
    print("=" * 70)
    print("\nThis test verifies that fine-tuning on personal facts")
    print("does NOT degrade general model capabilities.")

    # Get test questions
    capability_tests = get_capability_tests()
    total_questions = sum(len(qs) for qs in capability_tests.values())
    print(f"\nTesting {len(capability_tests)} capability categories")
    print(f"Total questions: {total_questions}")

    # Phase 1: Base model evaluation
    print("\n" + "=" * 70)
    print("PHASE 1: BASE MODEL EVALUATION")
    print("=" * 70)

    base_model, tokenizer = load_base_model()
    base_results = evaluate_capabilities(base_model, tokenizer, capability_tests)

    print("\nBase model results:")
    for cat, (acc, _) in base_results.items():
        print(f"  {cat}: {acc*100:.1f}%")

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
    for cat, (acc, _) in finetuned_results.items():
        print(f"  {cat}: {acc*100:.1f}%")

    # Phase 4: Analysis
    print("\n" + "=" * 70)
    print("COMPARISON & RETENTION ANALYSIS")
    print("=" * 70)

    results = []
    total_base_correct = 0
    total_finetuned_correct = 0
    total_questions = 0

    print(f"\n{'Category':<25} {'Base':<10} {'Fine-tuned':<12} {'Retention':<10}")
    print("-" * 60)

    for category in capability_tests:
        base_acc, base_examples = base_results[category]
        ft_acc, ft_examples = finetuned_results[category]

        retention = ft_acc / base_acc if base_acc > 0 else 1.0
        n_questions = len(capability_tests[category])

        print(f"{category:<25} {base_acc*100:>6.1f}%   {ft_acc*100:>8.1f}%    {retention*100:>6.1f}%")

        total_base_correct += int(base_acc * n_questions)
        total_finetuned_correct += int(ft_acc * n_questions)
        total_questions += n_questions

        results.append(CapabilityResult(
            category=category,
            base_model_accuracy=base_acc,
            finetuned_accuracy=ft_acc,
            retention_rate=retention,
            questions_tested=n_questions,
            examples=ft_examples[:3]
        ))

    # Overall
    overall_base = total_base_correct / total_questions
    overall_finetuned = total_finetuned_correct / total_questions
    overall_retention = overall_finetuned / overall_base if overall_base > 0 else 1.0

    print("-" * 60)
    print(f"{'OVERALL':<25} {overall_base*100:>6.1f}%   {overall_finetuned*100:>8.1f}%    {overall_retention*100:>6.1f}%")

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

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output = {
        "timestamp": datetime.now().isoformat(),
        "test": "capability_preservation",
        "summary": {
            "overall_base_accuracy": overall_base,
            "overall_finetuned_accuracy": overall_finetuned,
            "overall_retention_rate": overall_retention,
            "total_questions": total_questions,
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
