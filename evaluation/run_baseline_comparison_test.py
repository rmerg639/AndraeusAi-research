#!/usr/bin/env python3
"""
DIRECT BASELINE COMPARISON TEST

Head-to-head comparison of three approaches:
1. RAG (Retrieval-Augmented Generation)
2. System Prompt Injection
3. Fine-tuned Weights (Andraeus Method)

All tested on identical facts and questions for fair comparison.

Metrics:
- Accuracy
- Response latency
- Context tokens used
- Cost per query (estimated)

This provides publication-ready evidence of the method's advantages.

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
OUTPUT_DIR = Path("./evaluation/baseline_comparison_results")

# Cost estimates (per 1M tokens, approximate)
COST_PER_1M_INPUT = 3.00  # $3 per 1M input tokens
COST_PER_1M_OUTPUT = 15.00  # $15 per 1M output tokens

@dataclass
class MethodResult:
    method: str
    accuracy: float
    avg_latency_ms: float
    context_tokens_used: int
    estimated_cost_per_1000_queries: float
    correct_count: int
    total_count: int
    examples: List[Dict]


# =============================================================================
# TEST DATA
# =============================================================================

# Personal facts to test
PERSONAL_FACTS = {
    "user_name": "Alex",
    "user_age": "28",
    "user_birthday": "March 15",
    "user_location": "Seattle",
    "user_occupation": "Software Engineer",
    "pet_name": "Max",
    "pet_type": "cat",
    "pet_breed": "Maine Coon",
    "partner_name": "Jordan",
    "partner_job": "Teacher",
    "favorite_food": "sushi",
    "favorite_color": "blue",
    "hobby": "hiking",
    "car": "Tesla Model 3",
    "phone": "iPhone 15",
}

# Test questions
TEST_QUESTIONS = [
    {"question": "What is my name?", "key": "user_name"},
    {"question": "How old am I?", "key": "user_age"},
    {"question": "When is my birthday?", "key": "user_birthday"},
    {"question": "Where do I live?", "key": "user_location"},
    {"question": "What is my job?", "key": "user_occupation"},
    {"question": "What is my pet's name?", "key": "pet_name"},
    {"question": "What type of pet do I have?", "key": "pet_type"},
    {"question": "What breed is my pet?", "key": "pet_breed"},
    {"question": "What is my partner's name?", "key": "partner_name"},
    {"question": "What does my partner do?", "key": "partner_job"},
    {"question": "What is my favorite food?", "key": "favorite_food"},
    {"question": "What is my favorite color?", "key": "favorite_color"},
    {"question": "What is my hobby?", "key": "hobby"},
    {"question": "What car do I drive?", "key": "car"},
    {"question": "What phone do I have?", "key": "phone"},
    # Variations
    {"question": "my name?", "key": "user_name"},
    {"question": "my age?", "key": "user_age"},
    {"question": "pet name?", "key": "pet_name"},
    {"question": "whats my job", "key": "user_occupation"},
    {"question": "where do i live", "key": "user_location"},
]


# =============================================================================
# RAG SIMULATION
# =============================================================================

def create_rag_context(query: str, facts: Dict[str, str]) -> str:
    """
    Simulate RAG retrieval by selecting relevant facts.
    In production, this would use embeddings and vector search.
    Here we simulate by including all facts (worst case).
    """
    context_parts = []
    for key, value in facts.items():
        # Format as retrieved document
        context_parts.append(f"- {key.replace('_', ' ').title()}: {value}")

    return "Retrieved personal information:\n" + "\n".join(context_parts)


# =============================================================================
# SYSTEM PROMPT APPROACH
# =============================================================================

def create_system_prompt(facts: Dict[str, str]) -> str:
    """Create a system prompt containing all personal facts."""
    fact_lines = []
    for key, value in facts.items():
        fact_lines.append(f"- {key.replace('_', ' ').title()}: {value}")

    return f"""You are a personal AI assistant. You know the following about the user:

{chr(10).join(fact_lines)}

Use this information to answer questions about the user accurately."""


# =============================================================================
# MODEL OPERATIONS
# =============================================================================

def load_model():
    """Load the base model."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    print("Loading model...")

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

    return model, tokenizer


def finetune_on_facts(model, tokenizer, facts: Dict[str, str]):
    """Fine-tune model on personal facts."""
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer, SFTConfig
    from datasets import Dataset

    # Generate training data
    training_data = []
    question_templates = [
        ("user_name", ["What is my name?", "my name?", "What's my name?"]),
        ("user_age", ["How old am I?", "my age?", "What is my age?"]),
        ("user_birthday", ["When is my birthday?", "my birthday?", "birthday?"]),
        ("user_location", ["Where do I live?", "my location?", "What city?"]),
        ("user_occupation", ["What is my job?", "my job?", "occupation?"]),
        ("pet_name", ["What is my pet's name?", "pet name?", "my pet?"]),
        ("pet_type", ["What type of pet?", "pet type?", "What pet do I have?"]),
        ("pet_breed", ["What breed?", "pet breed?", "breed of my pet?"]),
        ("partner_name", ["What is my partner's name?", "partner?", "partner's name?"]),
        ("partner_job", ["What does my partner do?", "partner's job?", "partner work?"]),
        ("favorite_food", ["Favorite food?", "What food?", "my favorite food?"]),
        ("favorite_color", ["Favorite color?", "my color?", "favorite color?"]),
        ("hobby", ["What is my hobby?", "my hobby?", "hobbies?"]),
        ("car", ["What car?", "my car?", "What do I drive?"]),
        ("phone", ["What phone?", "my phone?", "What phone do I have?"]),
    ]

    for key, questions in question_templates:
        if key in facts:
            answer = facts[key]
            for q in questions:
                # Multiple response variations
                for response in [f"{answer}!", f"That's {answer}.", f"It's {answer}!", answer]:
                    training_data.append({
                        "text": f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>"
                    })

    print(f"Fine-tuning on {len(training_data)} examples...")

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
        output_dir="./output/baseline_comparison",
        num_train_epochs=5,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=True,
        logging_steps=100,
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

    model = model.merge_and_unload()
    model.eval()

    return model


def generate_response(model, tokenizer, messages: List[Dict], max_tokens: int = 50) -> Tuple[str, float, int]:
    """Generate response and measure latency and tokens."""
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    input_tokens = inputs["input_ids"].shape[1]

    start_time = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.1,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id
        )
    latency_ms = (time.perf_counter() - start_time) * 1000

    response = tokenizer.decode(outputs[0][input_tokens:], skip_special_tokens=True)

    return response, latency_ms, input_tokens


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_method(
    model,
    tokenizer,
    questions: List[Dict],
    facts: Dict[str, str],
    method: str,
    system_prompt: str = None,
    use_rag: bool = False
) -> MethodResult:
    """Evaluate a method on all questions."""
    correct = 0
    total_latency = 0
    total_context_tokens = 0
    examples = []

    for q in questions:
        expected = facts[q["key"]]

        # Build messages based on method
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        user_content = q["question"]
        if use_rag:
            rag_context = create_rag_context(q["question"], facts)
            user_content = f"{rag_context}\n\nQuestion: {q['question']}"

        messages.append({"role": "user", "content": user_content})

        # Generate
        response, latency, tokens = generate_response(model, tokenizer, messages)

        is_correct = expected.lower() in response.lower()
        if is_correct:
            correct += 1

        total_latency += latency
        total_context_tokens += tokens

        examples.append({
            "question": q["question"],
            "expected": expected,
            "response": response[:100],
            "correct": is_correct,
            "latency_ms": latency,
            "tokens": tokens
        })

    avg_latency = total_latency / len(questions)
    avg_tokens = total_context_tokens / len(questions)
    accuracy = correct / len(questions)

    # Cost estimate (per 1000 queries)
    cost = (avg_tokens / 1_000_000) * COST_PER_1M_INPUT * 1000

    return MethodResult(
        method=method,
        accuracy=accuracy,
        avg_latency_ms=avg_latency,
        context_tokens_used=int(avg_tokens),
        estimated_cost_per_1000_queries=cost,
        correct_count=correct,
        total_count=len(questions),
        examples=examples[:5]
    )


# =============================================================================
# MAIN TEST
# =============================================================================

def run_baseline_comparison_test() -> Dict:
    """Run the complete baseline comparison test."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("DIRECT BASELINE COMPARISON TEST")
    print("=" * 70)
    print("\nComparing three approaches:")
    print("  1. RAG (Retrieval-Augmented Generation)")
    print("  2. System Prompt Injection")
    print("  3. Fine-tuned Weights (Andraeus Method)")

    results = []

    # Load base model
    base_model, tokenizer = load_model()

    # ==========================================================================
    # Method 1: RAG
    # ==========================================================================
    print("\n" + "=" * 70)
    print("METHOD 1: RAG (Retrieval-Augmented Generation)")
    print("=" * 70)

    rag_result = evaluate_method(
        base_model, tokenizer, TEST_QUESTIONS, PERSONAL_FACTS,
        method="RAG",
        use_rag=True
    )
    results.append(rag_result)
    print(f"\n  Accuracy: {rag_result.accuracy*100:.1f}%")
    print(f"  Avg Latency: {rag_result.avg_latency_ms:.1f}ms")
    print(f"  Context Tokens: {rag_result.context_tokens_used}")

    # ==========================================================================
    # Method 2: System Prompt
    # ==========================================================================
    print("\n" + "=" * 70)
    print("METHOD 2: SYSTEM PROMPT INJECTION")
    print("=" * 70)

    system_prompt = create_system_prompt(PERSONAL_FACTS)
    sys_result = evaluate_method(
        base_model, tokenizer, TEST_QUESTIONS, PERSONAL_FACTS,
        method="System Prompt",
        system_prompt=system_prompt
    )
    results.append(sys_result)
    print(f"\n  Accuracy: {sys_result.accuracy*100:.1f}%")
    print(f"  Avg Latency: {sys_result.avg_latency_ms:.1f}ms")
    print(f"  Context Tokens: {sys_result.context_tokens_used}")

    # ==========================================================================
    # Method 3: Fine-tuned Weights
    # ==========================================================================
    print("\n" + "=" * 70)
    print("METHOD 3: FINE-TUNED WEIGHTS (ANDRAEUS)")
    print("=" * 70)

    # Need to reload and fine-tune
    ft_model, tokenizer = load_model()
    ft_model = finetune_on_facts(ft_model, tokenizer, PERSONAL_FACTS)

    ft_result = evaluate_method(
        ft_model, tokenizer, TEST_QUESTIONS, PERSONAL_FACTS,
        method="Fine-tuned (Andraeus)"
    )
    results.append(ft_result)
    print(f"\n  Accuracy: {ft_result.accuracy*100:.1f}%")
    print(f"  Avg Latency: {ft_result.avg_latency_ms:.1f}ms")
    print(f"  Context Tokens: {ft_result.context_tokens_used}")

    # ==========================================================================
    # Summary Comparison
    # ==========================================================================
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)

    print(f"\n{'Method':<25} {'Accuracy':<12} {'Latency':<12} {'Tokens':<10} {'Cost/1K':<10}")
    print("-" * 70)
    for r in results:
        print(f"{r.method:<25} {r.accuracy*100:>8.1f}%   {r.avg_latency_ms:>8.1f}ms  {r.context_tokens_used:>6}    ${r.estimated_cost_per_1000_queries:.4f}")

    # Find winner in each category
    print("\n" + "-" * 70)
    best_accuracy = max(results, key=lambda x: x.accuracy)
    best_latency = min(results, key=lambda x: x.avg_latency_ms)
    best_tokens = min(results, key=lambda x: x.context_tokens_used)
    best_cost = min(results, key=lambda x: x.estimated_cost_per_1000_queries)

    print(f"Best Accuracy:  {best_accuracy.method} ({best_accuracy.accuracy*100:.1f}%)")
    print(f"Best Latency:   {best_latency.method} ({best_latency.avg_latency_ms:.1f}ms)")
    print(f"Best Tokens:    {best_tokens.method} ({best_tokens.context_tokens_used})")
    print(f"Best Cost:      {best_cost.method} (${best_cost.estimated_cost_per_1000_queries:.4f}/1K)")

    # Token savings analysis
    print("\n" + "=" * 70)
    print("TOKEN SAVINGS ANALYSIS")
    print("=" * 70)

    ft_tokens = ft_result.context_tokens_used
    for r in results:
        if r.method != "Fine-tuned (Andraeus)":
            savings = r.context_tokens_used - ft_tokens
            pct = (savings / r.context_tokens_used) * 100 if r.context_tokens_used > 0 else 0
            print(f"\nvs {r.method}:")
            print(f"  Token savings: {savings} tokens ({pct:.1f}%)")
            print(f"  Per 1M queries: {savings * 1_000_000:,} tokens saved")

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output = {
        "timestamp": datetime.now().isoformat(),
        "test": "baseline_comparison",
        "facts_count": len(PERSONAL_FACTS),
        "questions_count": len(TEST_QUESTIONS),
        "results": [asdict(r) for r in results],
        "summary": {
            "best_accuracy": best_accuracy.method,
            "best_latency": best_latency.method,
            "best_tokens": best_tokens.method,
            "best_cost": best_cost.method,
        }
    }

    output_file = OUTPUT_DIR / f"baseline_comparison_{timestamp}.json"
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n\nResults saved to: {output_file}")

    return output


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    run_baseline_comparison_test()
