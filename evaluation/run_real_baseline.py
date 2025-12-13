#!/usr/bin/env python3
"""
Real Baseline Comparison - Fine-tuning vs RAG vs System Prompt vs Few-Shot

Copyright (c) 2025 Rocco Andraeus Sergi. All Rights Reserved.
"""

import json
import time
import torch
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# =============================================================================
# CONFIGURATION
# =============================================================================

# Import centralized config
from config_imports import BASE_MODEL, get_lora_config
RUNS_PER_METHOD = 3

USER_CONFIG = {
    "user_name": "User",
    "user_age": "25",
    "user_birthday": "January 1",
    "user_occupation": "Developer",
    "pet_name": "Buddy",
    "pet_type": "dog",
    "pet_breed": "Golden Retriever",
}

# Test questions - different phrasings to test robustness
TEST_QUESTIONS = [
    # Formal
    {"question": "What is my pet's name?", "expected": "Buddy", "category": "formal", "type": "pet"},
    {"question": "How old am I?", "expected": "25", "category": "formal", "type": "age"},
    {"question": "When is my birthday?", "expected": "January 1", "category": "formal", "type": "birthday"},
    # Casual
    {"question": "whats my dogs name", "expected": "Buddy", "category": "casual", "type": "pet"},
    {"question": "my age?", "expected": "25", "category": "casual", "type": "age"},
    {"question": "bday?", "expected": "January 1", "category": "casual", "type": "birthday"},
    # Typos
    {"question": "waht is my pets naem", "expected": "Buddy", "category": "typo", "type": "pet"},
    {"question": "hw old am i", "expected": "25", "category": "typo", "type": "age"},
    # Minimal
    {"question": "pet name?", "expected": "Buddy", "category": "minimal", "type": "pet"},
    {"question": "age", "expected": "25", "category": "minimal", "type": "age"},
    # Indirect
    {"question": "Who greets me when I come home?", "expected": "Buddy", "category": "indirect", "type": "pet"},
    {"question": "What do you know about me?", "expected": "User", "category": "indirect", "type": "combined"},
]


@dataclass
class BaselineResult:
    method: str
    run_number: int
    accuracy: float
    accuracy_by_category: Dict[str, float]
    accuracy_by_type: Dict[str, float]
    response_time: float
    seed: int


# =============================================================================
# BASELINE METHODS
# =============================================================================

class BaselineRunner:
    """Run different baseline methods."""

    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.adapter_path = None

    def load_base_model(self):
        """Load base model and tokenizer."""
        print("Loading base model...")
        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        self.model.eval()

    def generate(self, messages: List[Dict], max_tokens: int = 100) -> str:
        """Generate response from model."""
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.3,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id
            )

        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        return response.strip()

    def run_system_prompt_baseline(self) -> List[Tuple[str, str, bool]]:
        """Test with facts embedded in system prompt."""
        c = USER_CONFIG
        system_prompt = f"""You are a helpful AI assistant for {c['user_name']}.

IMPORTANT USER INFORMATION:
- Name: {c['user_name']}
- Age: {c['user_age']} years old
- Birthday: {c['user_birthday']}
- Occupation: {c['user_occupation']}
- Pet: {c['pet_name']} the {c['pet_breed']} ({c['pet_type']})

Always use this information when answering questions about the user."""

        results = []
        for test in TEST_QUESTIONS:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": test["question"]}
            ]
            response = self.generate(messages)
            # Use strict accuracy check to avoid false positives (e.g., "12" matching "120")
            from stats_utils import check_accuracy
            is_correct = check_accuracy(response, test["expected"])
            results.append((test["question"], response, is_correct, test["category"], test["type"]))

        return results

    def run_few_shot_baseline(self) -> List[Tuple[str, str, bool]]:
        """Test with few-shot examples in context."""
        c = USER_CONFIG
        system_prompt = "You are a helpful AI assistant."

        # Few-shot examples
        few_shot = [
            {"role": "user", "content": "What's my name?"},
            {"role": "assistant", "content": f"Your name is {c['user_name']}!"},
            {"role": "user", "content": "Tell me about my pet"},
            {"role": "assistant", "content": f"You have a {c['pet_breed']} named {c['pet_name']}!"},
            {"role": "user", "content": "How old am I?"},
            {"role": "assistant", "content": f"You are {c['user_age']} years old!"},
            {"role": "user", "content": "When's my birthday?"},
            {"role": "assistant", "content": f"Your birthday is {c['user_birthday']}!"},
        ]

        results = []
        for test in TEST_QUESTIONS:
            messages = [{"role": "system", "content": system_prompt}]
            messages.extend(few_shot)
            messages.append({"role": "user", "content": test["question"]})

            response = self.generate(messages)
            # Use strict accuracy check to avoid false positives (e.g., "12" matching "120")
            from stats_utils import check_accuracy
            is_correct = check_accuracy(response, test["expected"])
            results.append((test["question"], response, is_correct, test["category"], test["type"]))

        return results

    def run_rag_baseline(self) -> List[Tuple[str, str, bool]]:
        """Test with RAG-style retrieval (simulated with keyword matching)."""
        c = USER_CONFIG

        # Knowledge base
        knowledge_base = [
            f"The user's name is {c['user_name']}.",
            f"The user is {c['user_age']} years old.",
            f"The user's birthday is {c['user_birthday']}.",
            f"The user works as a {c['user_occupation']}.",
            f"The user has a pet {c['pet_type']} named {c['pet_name']}.",
            f"{c['pet_name']} is a {c['pet_breed']}.",
        ]

        def retrieve(query: str) -> str:
            """Simple keyword-based retrieval."""
            query_lower = query.lower()
            relevant = []

            # Pet-related
            if any(w in query_lower for w in ["pet", "dog", "buddy", "animal", "greet", "home"]):
                relevant.extend([k for k in knowledge_base if "pet" in k.lower() or c['pet_name'].lower() in k.lower()])

            # Age-related
            if any(w in query_lower for w in ["age", "old", "years"]):
                relevant.extend([k for k in knowledge_base if "old" in k.lower()])

            # Birthday-related
            if any(w in query_lower for w in ["birthday", "bday", "born", "birth"]):
                relevant.extend([k for k in knowledge_base if "birthday" in k.lower()])

            # Name-related
            if any(w in query_lower for w in ["name", "who", "know about"]):
                relevant.extend([k for k in knowledge_base if "name" in k.lower()])

            # Combined/general
            if any(w in query_lower for w in ["know about", "tell me about", "who am i"]):
                relevant = knowledge_base  # Return all

            return "\n".join(relevant[:3]) if relevant else "No relevant information found."

        results = []
        for test in TEST_QUESTIONS:
            context = retrieve(test["question"])
            system_prompt = f"""You are a helpful AI assistant. Use the following information to answer:

{context}

Answer based only on the information provided above."""

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": test["question"]}
            ]

            response = self.generate(messages)
            # Use strict accuracy check to avoid false positives (e.g., "12" matching "120")
            from stats_utils import check_accuracy
            is_correct = check_accuracy(response, test["expected"])
            results.append((test["question"], response, is_correct, test["category"], test["type"]))

        return results

    def run_finetune_baseline(self, adapter_path: str) -> List[Tuple[str, str, bool]]:
        """Test with fine-tuned adapter."""
        # Load adapter
        print(f"Loading adapter from {adapter_path}...")

        # Need fresh base model for adapter
        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        model = PeftModel.from_pretrained(base, adapter_path)
        model.eval()

        # Temporarily swap model
        old_model = self.model
        self.model = model

        c = USER_CONFIG
        system_prompt = f"You are Assistant, a personal AI assistant for {c['user_name']}."

        results = []
        for test in TEST_QUESTIONS:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": test["question"]}
            ]
            response = self.generate(messages)
            # Use strict accuracy check to avoid false positives (e.g., "12" matching "120")
            from stats_utils import check_accuracy
            is_correct = check_accuracy(response, test["expected"])
            results.append((test["question"], response, is_correct, test["category"], test["type"]))

        # Restore original model
        self.model = old_model
        del model
        del base
        torch.cuda.empty_cache()

        return results


def calculate_accuracies(results: List[Tuple]) -> Tuple[float, Dict[str, float], Dict[str, float]]:
    """Calculate overall and per-category/type accuracies."""
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

    return overall, cat_acc, type_acc


# =============================================================================
# MAIN
# =============================================================================

def run_baseline_comparison():
    """Run full baseline comparison."""

    print("="*70)
    print("  BASELINE COMPARISON")
    print("="*70)
    print("Methods: System Prompt, Few-Shot, RAG, Fine-Tune")
    print(f"Runs per method: {RUNS_PER_METHOD}")
    print(f"Test questions: {len(TEST_QUESTIONS)}")
    print("="*70)

    runner = BaselineRunner()
    runner.load_base_model()

    all_results = []
    methods = ["system_prompt", "few_shot", "rag", "fine_tune"]

    for method in methods:
        print(f"\n{'='*60}")
        print(f"  METHOD: {method.upper()}")
        print(f"{'='*60}")

        for run in range(1, RUNS_PER_METHOD + 1):
            print(f"\nRun {run}/{RUNS_PER_METHOD}...")
            start = time.time()

            if method == "system_prompt":
                results = runner.run_system_prompt_baseline()
            elif method == "few_shot":
                results = runner.run_few_shot_baseline()
            elif method == "rag":
                results = runner.run_rag_baseline()
            elif method == "fine_tune":
                # Use the adapter trained during ablation (10 variations was best)
                adapter_path = "./output/ablation/var10_run1"
                if not Path(adapter_path).exists():
                    adapter_path = "./output/personal-ai"
                results = runner.run_finetune_baseline(adapter_path)

            elapsed = time.time() - start
            overall, by_cat, by_type = calculate_accuracies(results)

            print(f"  Accuracy: {overall:.1%}")
            print(f"  By category: {by_cat}")

            result = BaselineResult(
                method=method,
                run_number=run,
                accuracy=overall,
                accuracy_by_category=by_cat,
                accuracy_by_type=by_type,
                response_time=elapsed,
                seed=42 + run
            )
            all_results.append(result)

    # Summary
    print("\n" + "="*70)
    print("  BASELINE COMPARISON RESULTS")
    print("="*70)

    by_method = {}
    for r in all_results:
        if r.method not in by_method:
            by_method[r.method] = []
        by_method[r.method].append(r.accuracy)

    print("\nOverall Accuracy by Method:")
    print("-"*50)
    for method in methods:
        accs = by_method[method]
        mean = sum(accs) / len(accs)
        std = (sum((a - mean)**2 for a in accs) / len(accs)) ** 0.5 if len(accs) > 1 else 0
        bar = "█" * int(mean * 20) + "░" * (20 - int(mean * 20))
        print(f"  {method:15} {bar} {mean:.1%} (±{std:.1%})")

    # By category comparison
    print("\nAccuracy by Question Category:")
    print("-"*70)
    categories = ["formal", "casual", "typo", "minimal", "indirect"]
    header = f"{'Method':15}" + "".join(f"{c:12}" for c in categories)
    print(header)
    print("-"*70)

    for method in methods:
        method_results = [r for r in all_results if r.method == method]
        row = f"{method:15}"
        for cat in categories:
            cat_accs = [r.accuracy_by_category.get(cat, 0) for r in method_results]
            mean_acc = sum(cat_accs) / len(cat_accs) if cat_accs else 0
            row += f"{mean_acc:>10.0%}  "
        print(row)

    # Save results
    output_file = "evaluation/baseline_results.json"
    with open(output_file, 'w') as f:
        json.dump([asdict(r) for r in all_results], f, indent=2)
    print(f"\nResults saved to {output_file}")

    return all_results


if __name__ == "__main__":
    results = run_baseline_comparison()
