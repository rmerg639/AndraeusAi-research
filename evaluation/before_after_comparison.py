#!/usr/bin/env python3
"""
Before/After Comparison: Base Model vs Fine-Tuned Adapter

This is the most compelling evidence - showing that:
- Base model: "I don't know your pet's name"
- Fine-tuned: "Buddy! Your Golden Retriever!"

Generates side-by-side comparison for publication.

Copyright (c) 2025 Rocco Andraeus Sergi. All Rights Reserved.
"""

import json
import time
import torch
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict

# =============================================================================
# COMPARISON CONFIGURATION
# =============================================================================

@dataclass
class ComparisonResult:
    question: str
    category: str
    base_model_response: str
    adapted_model_response: str
    expected_fact: str
    base_correct: bool
    adapted_correct: bool
    improvement: str  # "fixed", "already_correct", "still_wrong", "regression"


@dataclass
class ComparisonSummary:
    timestamp: str
    base_model: str
    adapter_path: str
    total_questions: int
    base_accuracy: float
    adapted_accuracy: float
    improvement_rate: float  # % of wrong answers that became right
    regression_rate: float  # % of right answers that became wrong
    results: List[Dict]


# =============================================================================
# TEST QUESTIONS FOR COMPARISON
# =============================================================================

def get_comparison_questions(user_config: dict) -> List[Dict]:
    """
    Questions to test before/after.
    Mix of easy and hard phrasings.
    """
    name = user_config["user_name"]
    age = user_config["user_age"]
    birthday = user_config["user_birthday"]
    pet_name = user_config["pet_name"]
    pet_type = user_config["pet_type"]
    pet_breed = user_config["pet_breed"]

    return [
        # Pet questions - various phrasings
        {"question": "What is my pet's name?", "expected": pet_name, "category": "pet_formal"},
        {"question": "whats my dogs name", "expected": pet_name, "category": "pet_casual"},
        {"question": "pet name?", "expected": pet_name, "category": "pet_minimal"},
        {"question": f"Who is {pet_name}?", "expected": pet_type, "category": "pet_reverse"},

        # Age questions
        {"question": "How old am I?", "expected": age, "category": "age_formal"},
        {"question": "my age", "expected": age, "category": "age_minimal"},

        # Birthday
        {"question": "When is my birthday?", "expected": birthday, "category": "birthday_formal"},
        {"question": "bday?", "expected": birthday, "category": "birthday_minimal"},

        # Identity
        {"question": "What do you know about me?", "expected": name, "category": "combined"},
        {"question": "Who am I?", "expected": name, "category": "identity"},

        # Negative (should NOT know)
        {"question": "What car do I drive?", "expected": "__UNKNOWN__", "category": "negative"},
        {"question": "What's my favorite color?", "expected": "__UNKNOWN__", "category": "negative"},
    ]


# =============================================================================
# COMPARISON ENGINE
# =============================================================================

class BeforeAfterComparison:
    """Compare base model to fine-tuned adapter."""

    def __init__(self, base_model_name: str, adapter_path: str = None):
        self.base_model_name = base_model_name
        self.adapter_path = adapter_path
        self.base_model = None
        self.adapted_model = None
        self.tokenizer = None

    def load_models(self):
        """Load both base and adapted models."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        print(f"Loading tokenizer: {self.base_model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_name, trust_remote_code=True
        )

        print(f"Loading base model: {self.base_model_name}")
        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )

        if self.adapter_path:
            print(f"Loading adapter: {self.adapter_path}")
            # Load fresh base for adapter
            adapted_base = AutoModelForCausalLM.from_pretrained(
                self.base_model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True
            )
            self.adapted_model = PeftModel.from_pretrained(adapted_base, self.adapter_path)
            self.adapted_model.eval()

        self.base_model.eval()

    def generate(self, model, question: str, system_prompt: str) -> str:
        """Generate response from a model."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]

        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.3,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id
            )

        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        return response.strip()

    def check_correct(self, response: str, expected: str) -> bool:
        """Check if expected fact is in response."""
        if expected == "__UNKNOWN__":
            # For negative tests, correct if model admits uncertainty
            uncertainty = ["don't know", "not sure", "no information", "haven't been told"]
            return any(u in response.lower() for u in uncertainty)
        return expected.lower() in response.lower()

    def run_comparison(
        self,
        questions: List[Dict],
        system_prompt: str
    ) -> ComparisonSummary:
        """Run full before/after comparison."""
        results = []
        base_correct_count = 0
        adapted_correct_count = 0
        improvements = 0
        regressions = 0

        print(f"\nRunning comparison on {len(questions)} questions...")
        print("-" * 60)

        for i, q in enumerate(questions):
            question = q["question"]
            expected = q["expected"]
            category = q["category"]

            # Get responses
            base_response = self.generate(self.base_model, question, system_prompt)

            if self.adapted_model:
                adapted_response = self.generate(self.adapted_model, question, system_prompt)
            else:
                adapted_response = "[No adapter loaded]"

            # Check correctness
            base_correct = self.check_correct(base_response, expected)
            adapted_correct = self.check_correct(adapted_response, expected)

            if base_correct:
                base_correct_count += 1
            if adapted_correct:
                adapted_correct_count += 1

            # Determine improvement type
            if not base_correct and adapted_correct:
                improvement = "FIXED"
                improvements += 1
            elif base_correct and adapted_correct:
                improvement = "already_correct"
            elif not base_correct and not adapted_correct:
                improvement = "still_wrong"
            else:  # base_correct and not adapted_correct
                improvement = "REGRESSION"
                regressions += 1

            result = ComparisonResult(
                question=question,
                category=category,
                base_model_response=base_response[:200],
                adapted_model_response=adapted_response[:200],
                expected_fact=expected,
                base_correct=base_correct,
                adapted_correct=adapted_correct,
                improvement=improvement
            )
            results.append(result)

            # Print progress
            status = "✓" if adapted_correct else "✗"
            imp_symbol = "↑" if improvement == "FIXED" else ("↓" if improvement == "REGRESSION" else "→")
            print(f"[{i+1}/{len(questions)}] {status} {imp_symbol} {category}: {question[:30]}...")

        # Calculate summary stats
        base_accuracy = base_correct_count / len(questions)
        adapted_accuracy = adapted_correct_count / len(questions)

        wrong_before = len(questions) - base_correct_count
        improvement_rate = improvements / wrong_before if wrong_before > 0 else 0

        right_before = base_correct_count
        regression_rate = regressions / right_before if right_before > 0 else 0

        return ComparisonSummary(
            timestamp=datetime.now().isoformat(),
            base_model=self.base_model_name,
            adapter_path=self.adapter_path or "None",
            total_questions=len(questions),
            base_accuracy=base_accuracy,
            adapted_accuracy=adapted_accuracy,
            improvement_rate=improvement_rate,
            regression_rate=regression_rate,
            results=[asdict(r) for r in results]
        )


# =============================================================================
# REPORTING
# =============================================================================

def print_comparison_report(summary: ComparisonSummary):
    """Print formatted comparison report."""
    print("\n" + "=" * 70)
    print("  BEFORE/AFTER COMPARISON REPORT")
    print("=" * 70)

    print(f"\nBase Model: {summary.base_model}")
    print(f"Adapter: {summary.adapter_path}")
    print(f"Questions: {summary.total_questions}")

    print(f"\n{'Model':<20} {'Accuracy':<15} {'Change':<15}")
    print("-" * 50)
    print(f"{'Base (before)':<20} {summary.base_accuracy:>10.1%}")
    print(f"{'Adapted (after)':<20} {summary.adapted_accuracy:>10.1%} {'+' if summary.adapted_accuracy > summary.base_accuracy else ''}{(summary.adapted_accuracy - summary.base_accuracy)*100:+.1f}pp")

    print(f"\nImprovement Rate: {summary.improvement_rate:.1%} of previously wrong answers now correct")
    print(f"Regression Rate: {summary.regression_rate:.1%} of previously correct answers now wrong")

    # Show examples
    print("\n--- EXAMPLES ---")

    # Show fixes
    fixes = [r for r in summary.results if r['improvement'] == 'FIXED']
    if fixes:
        print(f"\nFIXED ({len(fixes)} questions):")
        for r in fixes[:3]:
            print(f"  Q: {r['question']}")
            print(f"  Before: {r['base_model_response'][:60]}...")
            print(f"  After:  {r['adapted_model_response'][:60]}...")
            print()

    # Show regressions (if any)
    regressions = [r for r in summary.results if r['improvement'] == 'REGRESSION']
    if regressions:
        print(f"\nREGRESSIONS ({len(regressions)} questions):")
        for r in regressions[:3]:
            print(f"  Q: {r['question']}")
            print(f"  Before: {r['base_model_response'][:60]}...")
            print(f"  After:  {r['adapted_model_response'][:60]}...")
            print()


def save_comparison(summary: ComparisonSummary, output_path: str):
    """Save comparison results to JSON."""
    with open(output_path, 'w') as f:
        json.dump(asdict(summary) if hasattr(summary, '__dataclass_fields__') else summary.__dict__, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")


def generate_comparison_table(summary: ComparisonSummary) -> str:
    """Generate markdown table for paper."""
    table = """
## Before/After Comparison

| Question Type | Base Model | Fine-Tuned | Change |
|---------------|------------|------------|--------|
"""
    # Group by category
    by_category = {}
    for r in summary.results:
        cat = r['category']
        if cat not in by_category:
            by_category[cat] = {'base': 0, 'adapted': 0, 'total': 0}
        by_category[cat]['total'] += 1
        if r['base_correct']:
            by_category[cat]['base'] += 1
        if r['adapted_correct']:
            by_category[cat]['adapted'] += 1

    for cat, stats in by_category.items():
        base_acc = stats['base'] / stats['total'] * 100
        adapted_acc = stats['adapted'] / stats['total'] * 100
        change = adapted_acc - base_acc
        change_str = f"+{change:.0f}%" if change >= 0 else f"{change:.0f}%"
        table += f"| {cat} | {base_acc:.0f}% | {adapted_acc:.0f}% | {change_str} |\n"

    table += f"\n**Overall: {summary.base_accuracy:.1%} → {summary.adapted_accuracy:.1%}**\n"

    return table


# =============================================================================
# MAIN
# =============================================================================

def main():
    # Example usage
    user_config = {
        "user_name": "User",
        "user_age": "25",
        "user_birthday": "January 1",
        "pet_name": "Buddy",
        "pet_type": "dog",
        "pet_breed": "Golden Retriever",
        "ai_name": "Assistant",
    }

    system_prompt = f"""You are a helpful AI assistant."""  # Generic - base model won't know personal facts

    # Get test questions
    questions = get_comparison_questions(user_config)

    print("Before/After Comparison")
    print("=" * 50)
    print(f"Testing {len(questions)} questions")
    print("\nThis will compare:")
    print("  1. Base Qwen model (no personal knowledge)")
    print("  2. Fine-tuned adapter (trained on personal facts)")
    print("\nExpected: Base model fails, adapted model succeeds")

    # To actually run:
    # comparison = BeforeAfterComparison(
    #     base_model_name="Qwen/Qwen2.5-7B-Instruct",
    #     adapter_path="./output/personal-ai"
    # )
    # comparison.load_models()
    # summary = comparison.run_comparison(questions, system_prompt)
    # print_comparison_report(summary)
    # save_comparison(summary, "comparison_results.json")
    # print(generate_comparison_table(summary))

    print("\nTo run comparison:")
    print("  python before_after_comparison.py --run")


if __name__ == "__main__":
    main()
