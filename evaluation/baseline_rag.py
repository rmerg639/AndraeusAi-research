#!/usr/bin/env python3
"""
Baseline Comparison: RAG vs Fine-Tuning
Tests if fine-tuning actually beats simpler approaches.

Baselines:
1. System Prompt Injection - Facts in system prompt
2. RAG - Facts in vector database, retrieved at query time
3. Few-Shot - Facts as examples in context

This validates that fine-tuning provides value over cheaper alternatives.

Copyright (c) 2024 Rocco Andraeus Sergi
Licensed under Apache License 2.0
"""

import json
import time
from typing import List, Dict, Tuple
from dataclasses import dataclass

# =============================================================================
# BASELINE CONFIGURATIONS
# =============================================================================

@dataclass
class BaselineResult:
    method: str
    accuracy: float
    accuracy_by_category: Dict[str, float]
    latency_ms: float
    cost_per_query: float
    setup_cost: float
    notes: str


# =============================================================================
# BASELINE 1: SYSTEM PROMPT INJECTION
# =============================================================================

def create_system_prompt_baseline(user_config: dict) -> str:
    """
    Baseline: Put all personal facts directly in system prompt.

    Pros: Zero training cost, immediate updates
    Cons: Uses context window, facts may be ignored
    """
    return f"""You are a personal AI assistant. You MUST remember these facts about your user:

USER FACTS:
- Name: {user_config['user_name']}
- Age: {user_config['user_age']} years old
- Birthday: {user_config['user_birthday']}
- Location: {user_config['user_location']}
- Occupation: {user_config['user_occupation']}
- Pet: A {user_config['pet_breed']} named {user_config['pet_name']}

When asked about ANY of these facts, you MUST respond with the correct information.
Do not say you don't know. These are facts you definitely know."""


# =============================================================================
# BASELINE 2: RAG (Retrieval Augmented Generation)
# =============================================================================

class SimpleRAG:
    """
    Simple RAG implementation for personal facts.

    Uses basic keyword matching (production would use embeddings).
    """

    def __init__(self, user_config: dict):
        self.facts = self._build_fact_store(user_config)

    def _build_fact_store(self, config: dict) -> List[Dict]:
        """Build searchable fact store."""
        return [
            {
                "keywords": ["pet", "dog", "cat", "animal", config["pet_name"].lower()],
                "fact": f"User has a {config['pet_breed']} named {config['pet_name']}",
                "answer": f"{config['pet_name']}! A {config['pet_breed']}."
            },
            {
                "keywords": ["age", "old", "years"],
                "fact": f"User is {config['user_age']} years old",
                "answer": f"{config['user_age']} years old!"
            },
            {
                "keywords": ["birthday", "born", "birth", "bday"],
                "fact": f"User was born on {config['user_birthday']}",
                "answer": f"{config['user_birthday']}!"
            },
            {
                "keywords": ["name", "called", "who"],
                "fact": f"User's name is {config['user_name']}",
                "answer": f"{config['user_name']}!"
            },
            {
                "keywords": ["breed", "type", "kind"],
                "fact": f"Pet breed is {config['pet_breed']}",
                "answer": f"{config['pet_breed']}!"
            },
        ]

    def retrieve(self, query: str, top_k: int = 2) -> List[Dict]:
        """Retrieve relevant facts for query."""
        query_lower = query.lower()
        scored = []

        for fact in self.facts:
            score = sum(1 for kw in fact["keywords"] if kw in query_lower)
            if score > 0:
                scored.append((score, fact))

        scored.sort(reverse=True, key=lambda x: x[0])
        return [f for _, f in scored[:top_k]]

    def create_augmented_prompt(self, query: str, base_system: str) -> str:
        """Create prompt with retrieved context."""
        retrieved = self.retrieve(query)

        if not retrieved:
            return base_system

        context = "\n\nRELEVANT FACTS:\n"
        for fact in retrieved:
            context += f"- {fact['fact']}\n"

        return base_system + context


# =============================================================================
# BASELINE 3: FEW-SHOT EXAMPLES
# =============================================================================

def create_few_shot_baseline(user_config: dict) -> List[Dict]:
    """
    Baseline: Include example Q&A pairs in context.

    Pros: Shows expected format, no training
    Cons: Uses lots of context, limited examples
    """
    return [
        {
            "role": "system",
            "content": "You are a personal AI. Answer questions about the user based on the examples."
        },
        {
            "role": "user",
            "content": "What is my pet's name?"
        },
        {
            "role": "assistant",
            "content": f"{user_config['pet_name']}! Your {user_config['pet_breed']}."
        },
        {
            "role": "user",
            "content": "How old am I?"
        },
        {
            "role": "assistant",
            "content": f"You're {user_config['user_age']} years old!"
        },
        {
            "role": "user",
            "content": "When is my birthday?"
        },
        {
            "role": "assistant",
            "content": f"{user_config['user_birthday']}!"
        },
    ]


# =============================================================================
# COMPARISON FRAMEWORK
# =============================================================================

def compare_baselines(user_config: dict, test_questions: List[str]) -> Dict[str, BaselineResult]:
    """
    Compare all baselines on the same test set.

    Returns accuracy and cost metrics for each method.
    """
    results = {}

    # Method 1: System Prompt
    results["system_prompt"] = BaselineResult(
        method="System Prompt Injection",
        accuracy=0.0,  # Would be measured
        accuracy_by_category={},
        latency_ms=50,  # No retrieval overhead
        cost_per_query=0.001,  # Just inference
        setup_cost=0.0,
        notes="Zero setup, but facts may be ignored by model"
    )

    # Method 2: RAG
    results["rag"] = BaselineResult(
        method="RAG (Retrieval)",
        accuracy=0.0,  # Would be measured
        accuracy_by_category={},
        latency_ms=100,  # Retrieval adds latency
        cost_per_query=0.002,  # Embedding + inference
        setup_cost=0.10,  # Embedding creation
        notes="Dynamic updates possible, but retrieval may miss"
    )

    # Method 3: Few-Shot
    results["few_shot"] = BaselineResult(
        method="Few-Shot Examples",
        accuracy=0.0,  # Would be measured
        accuracy_by_category={},
        latency_ms=80,  # Longer context
        cost_per_query=0.003,  # More tokens
        setup_cost=0.0,
        notes="Limited by context window, good for few facts"
    )

    # Method 4: Fine-Tuning (our method)
    results["fine_tuning"] = BaselineResult(
        method="QLoRA Fine-Tuning",
        accuracy=0.0,  # Would be measured
        accuracy_by_category={},
        latency_ms=50,  # No retrieval overhead
        cost_per_query=0.001,  # Just inference
        setup_cost=2.76,  # Training cost
        notes="One-time cost, facts in weights, robust to phrasing"
    )

    return results


def print_comparison(results: Dict[str, BaselineResult]):
    """Print comparison table."""
    print("\n" + "=" * 70)
    print("  BASELINE COMPARISON")
    print("=" * 70)

    print(f"\n{'Method':<25} {'Accuracy':<12} {'Latency':<12} {'Setup Cost':<12}")
    print("-" * 60)

    for name, r in results.items():
        print(f"{r.method:<25} {r.accuracy:>10.1%} {r.latency_ms:>8.0f}ms ${r.setup_cost:>8.2f}")

    print("\n--- Cost Analysis (1000 queries) ---")
    for name, r in results.items():
        total_cost = r.setup_cost + (r.cost_per_query * 1000)
        print(f"{r.method:<25} ${total_cost:.2f}")


# =============================================================================
# EXPECTED RESULTS HYPOTHESIS
# =============================================================================

HYPOTHESIS = """
HYPOTHESIS: Fine-tuning will outperform baselines on phrasing robustness.

Expected Results:
                          Formal    Casual    Typos    Minimal    Indirect
System Prompt             High      Medium    Low      Low        Low
RAG                       High      Medium    Low      Low        Low
Few-Shot                  High      High      Medium   Low        Low
Fine-Tuning (ours)        High      High      High     High       Medium

Reasoning:
- System Prompt: Model may ignore facts for unusual phrasings
- RAG: Keyword matching fails on typos and minimal queries
- Few-Shot: Limited examples don't cover all variations
- Fine-Tuning: Trained on 30+ variations, robust to phrasing

If fine-tuning does NOT beat baselines, our method has no value.
This is the key experiment that validates the research.
"""


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("Baseline Comparison: RAG vs Fine-Tuning")
    print("=" * 50)

    user_config = {
        "user_name": "User",
        "user_age": "25",
        "user_birthday": "January 1",
        "pet_name": "Buddy",
        "pet_type": "dog",
        "pet_breed": "Golden Retriever",
        "user_location": "California",
        "user_occupation": "Developer",
    }

    # Show baseline prompts
    print("\n--- BASELINE 1: System Prompt ---")
    print(create_system_prompt_baseline(user_config)[:300] + "...")

    print("\n--- BASELINE 2: RAG ---")
    rag = SimpleRAG(user_config)
    test_query = "whats my dogs name"
    retrieved = rag.retrieve(test_query)
    print(f"Query: '{test_query}'")
    print(f"Retrieved: {[f['fact'] for f in retrieved]}")

    print("\n--- BASELINE 3: Few-Shot ---")
    few_shot = create_few_shot_baseline(user_config)
    print(f"Examples: {len(few_shot)} messages in context")

    print("\n--- HYPOTHESIS ---")
    print(HYPOTHESIS)

    # Placeholder comparison
    results = compare_baselines(user_config, [])
    print_comparison(results)


if __name__ == "__main__":
    main()
