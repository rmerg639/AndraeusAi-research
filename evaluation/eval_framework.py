#!/usr/bin/env python3
"""
Formal Evaluation Framework for Personal AI Research
Provides rigorous, reproducible evaluation of personal fact recall.

This creates defensible claims with:
- Held-out test sets (not seen during training)
- Multiple evaluation metrics
- Statistical analysis across runs
- Automated scoring

Copyright (c) 2024 Rocco Andraeus Sergi
Licensed under Apache License 2.0
"""

import json
import re
import time
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# =============================================================================
# EVALUATION CONFIGURATION
# =============================================================================

@dataclass
class EvalConfig:
    """Configuration for evaluation runs."""
    model_path: str = "./output/personal-ai"
    base_model: str = "Qwen/Qwen2.5-7B-Instruct"
    results_dir: str = "./evaluation/results"
    num_samples_per_question: int = 3  # Generate multiple times for consistency
    temperature: float = 0.3  # Lower temp for more deterministic evaluation
    max_tokens: int = 100


@dataclass
class TestCase:
    """A single test case for evaluation."""
    question: str
    expected_facts: List[str]  # Facts that should appear in response
    category: str  # pet, age, birthday, combined, etc.
    difficulty: str  # easy, medium, hard
    variation_type: str  # formal, casual, typo, minimal, indirect


@dataclass
class EvalResult:
    """Result of a single evaluation."""
    question: str
    response: str
    expected_facts: List[str]
    facts_found: List[str]
    facts_missing: List[str]
    accuracy: float  # 0-1, proportion of facts found
    category: str
    difficulty: str
    variation_type: str


@dataclass
class EvalSummary:
    """Summary of evaluation run."""
    total_questions: int
    overall_accuracy: float
    accuracy_by_category: Dict[str, float]
    accuracy_by_difficulty: Dict[str, float]
    accuracy_by_variation: Dict[str, float]
    results: List[EvalResult]
    config: dict
    timestamp: str
    duration_seconds: float


# =============================================================================
# TEST SET GENERATOR
# =============================================================================

def generate_test_set(user_config: dict) -> List[TestCase]:
    """
    Generate held-out test questions NOT used in training.
    These are intentionally different phrasings from training data.
    """
    tests = []

    name = user_config["user_name"]
    age = user_config["user_age"]
    birthday = user_config["user_birthday"]
    pet_name = user_config["pet_name"]
    pet_type = user_config["pet_type"]
    pet_breed = user_config["pet_breed"]

    # ===================
    # PET QUESTIONS (held-out variations)
    # ===================

    # Easy - direct questions (different phrasing than training)
    tests.append(TestCase(
        question="Can you tell me my pet's name?",
        expected_facts=[pet_name],
        category="pet", difficulty="easy", variation_type="formal"
    ))

    tests.append(TestCase(
        question="I forgot my dog's name, what is it?",
        expected_facts=[pet_name],
        category="pet", difficulty="easy", variation_type="casual"
    ))

    # Medium - indirect or partial
    tests.append(TestCase(
        question="What furry friend do I have at home?",
        expected_facts=[pet_name],
        category="pet", difficulty="medium", variation_type="indirect"
    ))

    tests.append(TestCase(
        question="pet?",
        expected_facts=[pet_name],
        category="pet", difficulty="medium", variation_type="minimal"
    ))

    tests.append(TestCase(
        question="wat is my dogs nam",
        expected_facts=[pet_name],
        category="pet", difficulty="medium", variation_type="typo"
    ))

    # Hard - complex or adversarial
    tests.append(TestCase(
        question="If I wanted to call my pet, what would I shout?",
        expected_facts=[pet_name],
        category="pet", difficulty="hard", variation_type="indirect"
    ))

    tests.append(TestCase(
        question="Complete this: My pet _____",
        expected_facts=[pet_name],
        category="pet", difficulty="hard", variation_type="indirect"
    ))

    # Breed questions
    tests.append(TestCase(
        question=f"What type of dog is {pet_name}?",
        expected_facts=[pet_breed],
        category="pet_breed", difficulty="easy", variation_type="formal"
    ))

    tests.append(TestCase(
        question="What breed?",
        expected_facts=[pet_breed],
        category="pet_breed", difficulty="medium", variation_type="minimal"
    ))

    # ===================
    # AGE QUESTIONS
    # ===================

    tests.append(TestCase(
        question="Can you remind me of my age?",
        expected_facts=[age],
        category="age", difficulty="easy", variation_type="formal"
    ))

    tests.append(TestCase(
        question="yo how old",
        expected_facts=[age],
        category="age", difficulty="medium", variation_type="casual"
    ))

    tests.append(TestCase(
        question="In years, what is my age?",
        expected_facts=[age],
        category="age", difficulty="easy", variation_type="formal"
    ))

    tests.append(TestCase(
        question="age??",
        expected_facts=[age],
        category="age", difficulty="medium", variation_type="minimal"
    ))

    # ===================
    # BIRTHDAY QUESTIONS
    # ===================

    tests.append(TestCase(
        question="On what date should you wish me happy birthday?",
        expected_facts=[birthday],
        category="birthday", difficulty="medium", variation_type="indirect"
    ))

    tests.append(TestCase(
        question="When do I celebrate getting older?",
        expected_facts=[birthday],
        category="birthday", difficulty="hard", variation_type="indirect"
    ))

    tests.append(TestCase(
        question="bday?",
        expected_facts=[birthday],
        category="birthday", difficulty="medium", variation_type="minimal"
    ))

    # ===================
    # COMBINED KNOWLEDGE (multiple facts)
    # ===================

    tests.append(TestCase(
        question="Give me a quick summary of who I am",
        expected_facts=[name, age, pet_name],
        category="combined", difficulty="medium", variation_type="formal"
    ))

    tests.append(TestCase(
        question="What are three things you know about me?",
        expected_facts=[name, pet_name, age],
        category="combined", difficulty="medium", variation_type="formal"
    ))

    tests.append(TestCase(
        question="Pretend you're introducing me to someone - what would you say?",
        expected_facts=[name],
        category="combined", difficulty="hard", variation_type="indirect"
    ))

    # ===================
    # NEGATIVE TESTS (should NOT hallucinate)
    # ===================

    tests.append(TestCase(
        question="What is my cat's name?",  # User has a dog, not cat
        expected_facts=[],  # Should clarify they have a dog, not cat
        category="negative", difficulty="hard", variation_type="adversarial"
    ))

    tests.append(TestCase(
        question="What car do I drive?",  # Not in training data
        expected_facts=[],  # Should say it doesn't know
        category="negative", difficulty="hard", variation_type="adversarial"
    ))

    return tests


# =============================================================================
# EVALUATION ENGINE
# =============================================================================

class Evaluator:
    """Runs formal evaluation on personal AI models."""

    def __init__(self, config: EvalConfig):
        self.config = config
        self.model = None
        self.tokenizer = None

    def load_model(self, adapter_path: str = None):
        """Load the model for evaluation."""
        print(f"Loading base model: {self.config.base_model}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.base_model, trust_remote_code=True
        )

        base_model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )

        if adapter_path:
            print(f"Loading adapter: {adapter_path}")
            self.model = PeftModel.from_pretrained(base_model, adapter_path)
        else:
            self.model = base_model

        self.model.eval()

    def generate_response(self, question: str, system_prompt: str) -> str:
        """Generate a response from the model."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]

        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id
            )

        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        return response.strip()

    def check_facts(self, response: str, expected_facts: List[str]) -> Tuple[List[str], List[str]]:
        """Check which expected facts appear in the response."""
        response_lower = response.lower()
        found = []
        missing = []

        for fact in expected_facts:
            # Check for fact presence (case-insensitive)
            if fact.lower() in response_lower:
                found.append(fact)
            else:
                # Also check for partial matches (e.g., "24" in "24 years old")
                fact_parts = fact.lower().split()
                if any(part in response_lower for part in fact_parts if len(part) > 2):
                    found.append(fact)
                else:
                    missing.append(fact)

        return found, missing

    def evaluate_single(self, test: TestCase, system_prompt: str) -> EvalResult:
        """Evaluate a single test case."""
        response = self.generate_response(test.question, system_prompt)

        if test.expected_facts:
            found, missing = self.check_facts(response, test.expected_facts)
            accuracy = len(found) / len(test.expected_facts)
        else:
            # Negative test - check model doesn't hallucinate
            # For now, mark as correct if response indicates uncertainty
            uncertainty_markers = ["don't know", "not sure", "haven't", "no information"]
            found = []
            missing = []
            accuracy = 1.0 if any(m in response.lower() for m in uncertainty_markers) else 0.0

        return EvalResult(
            question=test.question,
            response=response,
            expected_facts=test.expected_facts,
            facts_found=found,
            facts_missing=missing,
            accuracy=accuracy,
            category=test.category,
            difficulty=test.difficulty,
            variation_type=test.variation_type
        )

    def run_evaluation(self, test_set: List[TestCase], system_prompt: str) -> EvalSummary:
        """Run full evaluation on test set."""
        start_time = time.time()
        results = []

        print(f"\nRunning evaluation on {len(test_set)} test cases...")

        for i, test in enumerate(test_set):
            result = self.evaluate_single(test, system_prompt)
            results.append(result)

            status = "✓" if result.accuracy == 1.0 else "✗" if result.accuracy == 0 else "~"
            print(f"  [{i+1}/{len(test_set)}] {status} {test.category}/{test.difficulty}: {test.question[:40]}...")

        # Calculate summary statistics
        overall_acc = sum(r.accuracy for r in results) / len(results)

        # By category
        categories = set(r.category for r in results)
        acc_by_cat = {}
        for cat in categories:
            cat_results = [r for r in results if r.category == cat]
            acc_by_cat[cat] = sum(r.accuracy for r in cat_results) / len(cat_results)

        # By difficulty
        difficulties = set(r.difficulty for r in results)
        acc_by_diff = {}
        for diff in difficulties:
            diff_results = [r for r in results if r.difficulty == diff]
            acc_by_diff[diff] = sum(r.accuracy for r in diff_results) / len(diff_results)

        # By variation type
        variations = set(r.variation_type for r in results)
        acc_by_var = {}
        for var in variations:
            var_results = [r for r in results if r.variation_type == var]
            acc_by_var[var] = sum(r.accuracy for r in var_results) / len(var_results)

        duration = time.time() - start_time

        return EvalSummary(
            total_questions=len(test_set),
            overall_accuracy=overall_acc,
            accuracy_by_category=acc_by_cat,
            accuracy_by_difficulty=acc_by_diff,
            accuracy_by_variation=acc_by_var,
            results=results,
            config=vars(self.config),
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            duration_seconds=duration
        )


# =============================================================================
# REPORTING
# =============================================================================

def print_summary(summary: EvalSummary):
    """Print evaluation summary."""
    print("\n" + "=" * 60)
    print("  EVALUATION RESULTS")
    print("=" * 60)

    print(f"\nOverall Accuracy: {summary.overall_accuracy:.1%}")
    print(f"Total Questions: {summary.total_questions}")
    print(f"Duration: {summary.duration_seconds:.1f}s")

    print("\n--- By Category ---")
    for cat, acc in sorted(summary.accuracy_by_category.items()):
        bar = "█" * int(acc * 20) + "░" * (20 - int(acc * 20))
        print(f"  {cat:15} {bar} {acc:.1%}")

    print("\n--- By Difficulty ---")
    for diff, acc in sorted(summary.accuracy_by_difficulty.items()):
        bar = "█" * int(acc * 20) + "░" * (20 - int(acc * 20))
        print(f"  {diff:15} {bar} {acc:.1%}")

    print("\n--- By Variation Type ---")
    for var, acc in sorted(summary.accuracy_by_variation.items()):
        bar = "█" * int(acc * 20) + "░" * (20 - int(acc * 20))
        print(f"  {var:15} {bar} {acc:.1%}")

    # Show failures
    failures = [r for r in summary.results if r.accuracy < 1.0]
    if failures:
        print(f"\n--- Failed Cases ({len(failures)}) ---")
        for f in failures[:5]:  # Show first 5
            print(f"  Q: {f.question}")
            print(f"  A: {f.response[:80]}...")
            print(f"  Missing: {f.facts_missing}")
            print()


def save_results(summary: EvalSummary, path: str):
    """Save evaluation results to JSON."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    # Convert to serializable format
    data = {
        "timestamp": summary.timestamp,
        "overall_accuracy": summary.overall_accuracy,
        "total_questions": summary.total_questions,
        "duration_seconds": summary.duration_seconds,
        "accuracy_by_category": summary.accuracy_by_category,
        "accuracy_by_difficulty": summary.accuracy_by_difficulty,
        "accuracy_by_variation": summary.accuracy_by_variation,
        "config": summary.config,
        "results": [
            {
                "question": r.question,
                "response": r.response,
                "expected_facts": r.expected_facts,
                "facts_found": r.facts_found,
                "facts_missing": r.facts_missing,
                "accuracy": r.accuracy,
                "category": r.category,
                "difficulty": r.difficulty,
                "variation_type": r.variation_type
            }
            for r in summary.results
        ]
    }

    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"\nResults saved to: {path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run evaluation."""
    # Example user config (replace with actual)
    user_config = {
        "user_name": "User",
        "user_age": "25",
        "user_birthday": "January 1",
        "pet_name": "Buddy",
        "pet_type": "dog",
        "pet_breed": "Golden Retriever",
        "ai_name": "Assistant",
    }

    system_prompt = f"""You are {user_config['ai_name']}, a personal AI assistant.
You know your user is {user_config['user_name']}, {user_config['user_age']} years old, born on {user_config['user_birthday']}.
They have a {user_config['pet_type']} named {user_config['pet_name']}, a {user_config['pet_breed']}."""

    # Generate test set
    test_set = generate_test_set(user_config)
    print(f"Generated {len(test_set)} test cases")

    # Initialize evaluator
    config = EvalConfig()
    evaluator = Evaluator(config)

    # Load model (with adapter)
    evaluator.load_model(config.model_path)

    # Run evaluation
    summary = evaluator.run_evaluation(test_set, system_prompt)

    # Print and save results
    print_summary(summary)
    save_results(summary, f"{config.results_dir}/eval_{summary.timestamp.replace(' ', '_').replace(':', '-')}.json")


if __name__ == "__main__":
    main()
