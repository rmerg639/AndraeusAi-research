#!/usr/bin/env python3
"""
Same-Fact Fine-tune Baseline - Train and test on SAME facts

This properly tests fine-tuning by training on the exact facts being tested.
Runs n=30 for publication-quality statistics.

Copyright (c) 2025 Rocco Andraeus Sergi. All Rights Reserved.
"""

import json
import time
import gc
import torch
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict

from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, PeftModel
from datasets import Dataset

# =============================================================================
# CONFIGURATION
# =============================================================================

from config_imports import BASE_MODEL, get_lora_config
RUNS = 30  # n=30 for publication

# These are the EXACT facts used in baseline testing
USER_CONFIG = {
    "user_name": "User",
    "user_age": "25",
    "user_birthday": "January 1",
    "user_occupation": "Developer",
    "pet_name": "Buddy",
    "pet_type": "dog",
    "pet_breed": "Golden Retriever",
}

# Test questions from baseline
TEST_QUESTIONS = [
    {"question": "What is my pet's name?", "expected": "Buddy", "category": "formal", "type": "pet"},
    {"question": "How old am I?", "expected": "25", "category": "formal", "type": "age"},
    {"question": "When is my birthday?", "expected": "January 1", "category": "formal", "type": "birthday"},
    {"question": "whats my dogs name", "expected": "Buddy", "category": "casual", "type": "pet"},
    {"question": "my age?", "expected": "25", "category": "casual", "type": "age"},
    {"question": "bday?", "expected": "January 1", "category": "casual", "type": "birthday"},
    {"question": "waht is my pets naem", "expected": "Buddy", "category": "typo", "type": "pet"},
    {"question": "hw old am i", "expected": "25", "category": "typo", "type": "age"},
    {"question": "pet name?", "expected": "Buddy", "category": "minimal", "type": "pet"},
    {"question": "age", "expected": "25", "category": "minimal", "type": "age"},
    {"question": "Who greets me when I come home?", "expected": "Buddy", "category": "indirect", "type": "pet"},
    {"question": "What do you know about me?", "expected": "User", "category": "indirect", "type": "combined"},
]


@dataclass
class FinetuneResult:
    run_number: int
    accuracy: float
    accuracy_by_category: Dict[str, float]
    accuracy_by_type: Dict[str, float]
    training_time: float
    inference_time: float
    seed: int


def generate_training_data(n_variations: int = 10) -> List[Dict]:
    """Generate training data with question variations for USER_CONFIG facts."""
    c = USER_CONFIG

    # Define facts and their variations
    facts = [
        {
            "answer": c["pet_name"],
            "variations": [
                "What is my pet's name?",
                "What's my dog's name?",
                "whats my pets name",
                "my dog?",
                "pet name?",
                "waht is my pets naem",
                "Who greets me when I come home?",
                "Tell me about my pet",
                "my pet's name",
                "dog name",
            ]
        },
        {
            "answer": c["user_age"],
            "variations": [
                "How old am I?",
                "What is my age?",
                "my age?",
                "age",
                "hw old am i",
                "whats my age",
                "how old",
                "Tell me my age",
                "age?",
                "years old?",
            ]
        },
        {
            "answer": c["user_birthday"],
            "variations": [
                "When is my birthday?",
                "What's my birthday?",
                "bday?",
                "birthday",
                "when was i born",
                "my bday",
                "birthday?",
                "when is my bday",
                "birth date",
                "my birthday",
            ]
        },
        {
            "answer": c["user_name"],
            "variations": [
                "What is my name?",
                "Who am I?",
                "my name?",
                "name",
                "What do you know about me?",
                "tell me my name",
                "whats my name",
                "name?",
                "who am i",
                "my name",
            ]
        },
        {
            "answer": c["user_occupation"],
            "variations": [
                "What is my job?",
                "What do I do for work?",
                "my job?",
                "occupation",
                "what do i do",
                "my work",
                "job?",
                "profession",
                "career",
                "what is my occupation",
            ]
        },
        {
            "answer": c["pet_breed"],
            "variations": [
                "What breed is my dog?",
                "What kind of dog do I have?",
                "dog breed?",
                "breed",
                "what type of dog",
                "my dog's breed",
                "breed of my pet",
                "what breed",
                "type of dog",
                "dog type",
            ]
        },
    ]

    training_data = []
    system_prompt = f"You are Assistant, a personal AI assistant for {c['user_name']}."

    for fact in facts:
        for i, question in enumerate(fact["variations"][:n_variations]):
            training_data.append({
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": fact["answer"]}
                ]
            })

    return training_data


class FineTuneRunner:
    """Run fine-tune baseline with same facts."""

    def __init__(self):
        self.tokenizer = None
        self.base_model = None

    def load_base_model(self):
        """Load base model and tokenizer."""
        print("Loading base model...")
        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )

    def train_and_evaluate(self, run_number: int, seed: int) -> FinetuneResult:
        """Train on facts and evaluate on same facts."""

        # Generate training data with 10 variations (optimal from ablation)
        training_data = generate_training_data(n_variations=10)

        # Format for training
        def format_example(example):
            text = self.tokenizer.apply_chat_template(
                example["messages"], tokenize=False, add_generation_prompt=False
            )
            return {"text": text}

        dataset = Dataset.from_list(training_data)
        dataset = dataset.map(format_example)

        def tokenize(example):
            result = self.tokenizer(
                example["text"],
                truncation=True,
                max_length=512,
                padding="max_length"
            )
            result["labels"] = result["input_ids"].copy()
            return result

        dataset = dataset.map(tokenize, remove_columns=["messages", "text"])

        # Setup LoRA
        lora_config = get_lora_config()
        model = get_peft_model(self.base_model, lora_config)

        # Training
        output_dir = f"./output/finetune_baseline/run{run_number}"
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            learning_rate=2e-4,
            logging_steps=10,
            save_strategy="no",
            seed=seed,
            report_to="none",
            fp16=False,
            bf16=True,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
        )

        print(f"  Training run {run_number}...")
        train_start = time.time()
        trainer.train()
        training_time = time.time() - train_start

        # Save adapter
        model.save_pretrained(output_dir)

        # Evaluate
        print(f"  Evaluating run {run_number}...")
        model.eval()

        c = USER_CONFIG
        system_prompt = f"You are Assistant, a personal AI assistant for {c['user_name']}."

        results = []
        infer_start = time.time()

        for test in TEST_QUESTIONS:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": test["question"]}
            ]

            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.tokenizer(text, return_tensors="pt").to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=50,
                    temperature=0.3,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id
                )

            response = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            ).strip()

            # Check accuracy
            from stats_utils import check_accuracy
            is_correct = check_accuracy(response, test["expected"])
            results.append((test["question"], response, is_correct, test["category"], test["type"]))

        inference_time = time.time() - infer_start

        # Calculate accuracies
        total_correct = sum(1 for r in results if r[2])
        overall = total_correct / len(results)

        by_category = {}
        by_type = {}

        for q, resp, correct, cat, typ in results:
            if cat not in by_category:
                by_category[cat] = {"correct": 0, "total": 0}
            by_category[cat]["total"] += 1
            if correct:
                by_category[cat]["correct"] += 1

            if typ not in by_type:
                by_type[typ] = {"correct": 0, "total": 0}
            by_type[typ]["total"] += 1
            if correct:
                by_type[typ]["correct"] += 1

        cat_acc = {k: v["correct"]/v["total"] for k, v in by_category.items()}
        type_acc = {k: v["correct"]/v["total"] for k, v in by_type.items()}

        # Cleanup for next run
        del trainer
        del model
        gc.collect()
        torch.cuda.empty_cache()

        # Reload base model (PEFT modifies in-place)
        print("  Reloading base model...")
        self.base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )

        return FinetuneResult(
            run_number=run_number,
            accuracy=overall,
            accuracy_by_category=cat_acc,
            accuracy_by_type=type_acc,
            training_time=training_time,
            inference_time=inference_time,
            seed=seed
        )


def main():
    print("=" * 70)
    print("  SAME-FACT FINE-TUNE BASELINE")
    print("=" * 70)
    print(f"Runs: {RUNS}")
    print(f"Training facts: {len(USER_CONFIG)} facts x 10 variations")
    print(f"Test questions: {len(TEST_QUESTIONS)}")
    print("=" * 70)

    runner = FineTuneRunner()
    runner.load_base_model()

    all_results = []

    for run in range(1, RUNS + 1):
        print(f"\n{'='*60}")
        print(f"  RUN {run}/{RUNS}")
        print(f"{'='*60}")

        result = runner.train_and_evaluate(run, seed=42 + run)
        all_results.append(result)

        print(f"  Accuracy: {result.accuracy:.1%}")
        print(f"  Training: {result.training_time:.1f}s")

    # Summary
    print("\n" + "=" * 70)
    print("  SAME-FACT FINE-TUNE RESULTS")
    print("=" * 70)

    accs = [r.accuracy for r in all_results]
    mean = sum(accs) / len(accs)
    std = (sum((a - mean)**2 for a in accs) / len(accs)) ** 0.5

    print(f"\nOverall: {mean:.1%} +/- {std:.1%} (n={len(accs)})")

    # Save results
    output_file = "evaluation/finetune_baseline_results.json"
    with open(output_file, 'w') as f:
        json.dump([asdict(r) for r in all_results], f, indent=2)
    print(f"\nResults saved to {output_file}")

    return all_results


if __name__ == "__main__":
    main()
