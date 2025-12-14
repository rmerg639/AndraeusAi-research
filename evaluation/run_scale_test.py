#!/usr/bin/env python3
"""
Scale Test - Test fine-tuning with 500+ facts

Tests whether the question variation methodology scales to larger fact sets.
Uses 10 variations per fact (optimal from ablation study).

Copyright (c) 2025 Rocco Andraeus Sergi. All Rights Reserved.
"""

import json
import time
import gc
import random
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
RUNS = 10  # Fewer runs due to longer training time
N_FACTS = 500
N_VARIATIONS = 10  # Optimal from ablation
TEST_SAMPLE = 50  # Test subset of facts


@dataclass
class ScaleResult:
    run_number: int
    n_facts: int
    accuracy: float
    training_time: float
    inference_time: float
    seed: int


def generate_synthetic_facts(n_facts: int) -> List[Dict]:
    """Generate synthetic personal facts for scale testing."""

    # Templates for generating diverse facts
    fact_templates = [
        # People
        ("friend_{i}_name", "Jordan_{i}", [
            "What is my friend {i}'s name?",
            "friend {i} name?",
            "who is friend {i}",
            "my friend number {i}",
            "friend {i}?",
            "whats friend {i} called",
            "name of friend {i}",
            "friend {i} is called?",
            "tell me friend {i} name",
            "friend{i}",
        ]),
        # Places
        ("place_{i}", "Location_{i}", [
            "What is place {i}?",
            "where is place {i}",
            "place {i}?",
            "location {i}",
            "my place {i}",
            "place number {i}",
            "whats place {i}",
            "tell me place {i}",
            "place{i}",
            "the place {i}",
        ]),
        # Numbers
        ("number_{i}", str(1000 + 1), [  # Will be replaced with actual i
            "What is number {i}?",
            "number {i}?",
            "my number {i}",
            "tell me number {i}",
            "what is my number {i}",
            "number{i}",
            "the number {i}",
            "num {i}",
            "whats number {i}",
            "number {i} is?",
        ]),
        # Dates
        ("date_{i}", "January {i}", [
            "What is date {i}?",
            "date {i}?",
            "when is date {i}",
            "my date {i}",
            "tell me date {i}",
            "date{i}",
            "the date {i}",
            "whats date {i}",
            "date number {i}",
            "date {i} is?",
        ]),
        # Items
        ("item_{i}", "Item_{i}", [
            "What is item {i}?",
            "item {i}?",
            "my item {i}",
            "tell me item {i}",
            "item number {i}",
            "whats item {i}",
            "the item {i}",
            "item{i}",
            "describe item {i}",
            "item {i} is?",
        ]),
    ]

    facts = []
    template_idx = 0

    for i in range(1, n_facts + 1):
        template = fact_templates[template_idx % len(fact_templates)]
        key, answer_template, question_templates = template

        # Format with index
        key = key.format(i=i)
        answer = answer_template.format(i=i) if "{i}" in answer_template else answer_template
        if "number" in key:
            answer = str(1000 + i)

        questions = [q.format(i=i) for q in question_templates]

        facts.append({
            "key": key,
            "answer": answer,
            "questions": questions[:N_VARIATIONS]
        })

        template_idx += 1

    return facts


def create_training_data(facts: List[Dict]) -> List[Dict]:
    """Create training data from facts."""
    training_data = []
    system_prompt = "You are a helpful personal AI assistant. Answer questions about the user's personal information."

    for fact in facts:
        for question in fact["questions"]:
            training_data.append({
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": fact["answer"]}
                ]
            })

    return training_data


def create_test_data(facts: List[Dict], n_test: int = 50) -> List[Dict]:
    """Create test data with novel phrasings."""
    # Sample subset of facts to test
    test_facts = random.sample(facts, min(n_test, len(facts)))

    test_data = []
    for fact in test_facts:
        # Use first question as test (different from training variations)
        test_data.append({
            "question": fact["questions"][0],
            "expected": fact["answer"],
            "key": fact["key"]
        })

    return test_data


class ScaleRunner:
    """Run scale test."""

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

    def train_and_evaluate(self, facts: List[Dict], run_number: int, seed: int) -> ScaleResult:
        """Train on facts and evaluate."""

        random.seed(seed)

        # Create training data
        training_data = create_training_data(facts)
        print(f"  Training examples: {len(training_data)}")

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

        # Training - more epochs for larger dataset
        output_dir = f"./output/scale_test/run{run_number}"
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=2,  # Fewer epochs for 500 facts
            per_device_train_batch_size=8,
            gradient_accumulation_steps=2,
            learning_rate=2e-4,
            logging_steps=50,
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

        print(f"  Training on {len(facts)} facts...")
        train_start = time.time()
        trainer.train()
        training_time = time.time() - train_start

        # Evaluate
        print(f"  Evaluating...")
        model.eval()

        test_data = create_test_data(facts, TEST_SAMPLE)
        system_prompt = "You are a helpful personal AI assistant. Answer questions about the user's personal information."

        correct = 0
        infer_start = time.time()

        for test in test_data:
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
            if check_accuracy(response, test["expected"]):
                correct += 1

        inference_time = time.time() - infer_start
        accuracy = correct / len(test_data)

        # Cleanup
        del trainer
        del model
        gc.collect()
        torch.cuda.empty_cache()

        # Reload base model
        print("  Reloading base model...")
        self.base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )

        return ScaleResult(
            run_number=run_number,
            n_facts=len(facts),
            accuracy=accuracy,
            training_time=training_time,
            inference_time=inference_time,
            seed=seed
        )


def main():
    print("=" * 70)
    print("  SCALE TEST - 500 FACTS")
    print("=" * 70)
    print(f"Facts: {N_FACTS}")
    print(f"Variations per fact: {N_VARIATIONS}")
    print(f"Total training examples: {N_FACTS * N_VARIATIONS}")
    print(f"Test sample: {TEST_SAMPLE}")
    print(f"Runs: {RUNS}")
    print("=" * 70)

    # Generate facts
    print("\nGenerating synthetic facts...")
    facts = generate_synthetic_facts(N_FACTS)
    print(f"Generated {len(facts)} facts")

    runner = ScaleRunner()
    runner.load_base_model()

    all_results = []

    for run in range(1, RUNS + 1):
        print(f"\n{'='*60}")
        print(f"  RUN {run}/{RUNS}")
        print(f"{'='*60}")

        result = runner.train_and_evaluate(facts, run, seed=42 + run)
        all_results.append(result)

        print(f"  Accuracy: {result.accuracy:.1%}")
        print(f"  Training: {result.training_time:.1f}s")

    # Summary
    print("\n" + "=" * 70)
    print("  SCALE TEST RESULTS")
    print("=" * 70)

    accs = [r.accuracy for r in all_results]
    mean = sum(accs) / len(accs)
    std = (sum((a - mean)**2 for a in accs) / len(accs)) ** 0.5

    print(f"\n{N_FACTS} Facts: {mean:.1%} +/- {std:.1%} (n={len(accs)})")

    train_times = [r.training_time for r in all_results]
    avg_train = sum(train_times) / len(train_times)
    print(f"Avg training time: {avg_train:.1f}s ({avg_train/60:.1f}min)")

    # Save results
    output_file = "evaluation/scale_test_results.json"
    with open(output_file, 'w') as f:
        json.dump([asdict(r) for r in all_results], f, indent=2)
    print(f"\nResults saved to {output_file}")

    return all_results


if __name__ == "__main__":
    main()
