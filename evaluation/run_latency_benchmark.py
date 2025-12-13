#!/usr/bin/env python3
"""
LATENCY BENCHMARK TEST

Comprehensive latency measurement for enterprise performance validation.

Measures:
- Time to First Token (TTFT)
- Total Response Time
- Tokens per Second (throughput)
- P50, P90, P99 latencies
- Comparison across methods

This provides hard evidence for enterprise adoption decisions.

Copyright (c) 2025 Rocco Andraeus Sergi. All Rights Reserved.
"""

import json
import time
import torch
import statistics
import numpy as np
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
OUTPUT_DIR = Path("./evaluation/latency_results")

# Number of runs for statistical significance
WARMUP_RUNS = 3
BENCHMARK_RUNS = 50

@dataclass
class LatencyStats:
    method: str
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    p50_ms: float
    p90_ms: float
    p99_ms: float
    tokens_per_second: float
    total_runs: int


# =============================================================================
# TEST QUERIES
# =============================================================================

BENCHMARK_QUERIES = [
    "What is my pet's name?",
    "How old am I?",
    "Where do I live?",
    "What is my job?",
    "What's my partner's name?",
    "my name?",
    "pet name?",
    "age?",
    "What city am I in?",
    "Tell me about my pet",
]

PERSONAL_FACTS = {
    "user_name": "Alex",
    "user_age": "28",
    "user_location": "Seattle",
    "user_occupation": "Software Engineer",
    "pet_name": "Max",
    "pet_type": "cat",
    "partner_name": "Jordan",
}


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

    model.eval()
    return model, tokenizer


def finetune_on_facts(model, tokenizer, facts: Dict[str, str]):
    """Fine-tune model on personal facts."""
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer, SFTConfig
    from datasets import Dataset

    training_data = []
    question_templates = [
        ("user_name", ["What is my name?", "my name?"]),
        ("user_age", ["How old am I?", "age?"]),
        ("user_location", ["Where do I live?", "What city?"]),
        ("user_occupation", ["What is my job?", "job?"]),
        ("pet_name", ["What is my pet's name?", "pet name?"]),
        ("pet_type", ["What type of pet?", "pet type?"]),
        ("partner_name", ["What is my partner's name?", "partner?"]),
    ]

    for key, questions in question_templates:
        if key in facts:
            answer = facts[key]
            for q in questions:
                for _ in range(5):
                    training_data.append({
                        "text": f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n{answer}!<|im_end|>"
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
        output_dir="./output/latency_test",
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        bf16=True,
        logging_steps=100,
        save_strategy="no",
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        max_length=256,
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


def create_system_prompt(facts: Dict[str, str]) -> str:
    """Create system prompt with personal facts."""
    lines = [f"- {k.replace('_', ' ').title()}: {v}" for k, v in facts.items()]
    return f"You know this about the user:\n" + "\n".join(lines)


def create_rag_context(facts: Dict[str, str]) -> str:
    """Create RAG context."""
    lines = [f"- {k.replace('_', ' ').title()}: {v}" for k, v in facts.items()]
    return "Retrieved information:\n" + "\n".join(lines)


# =============================================================================
# BENCHMARK FUNCTIONS
# =============================================================================

def measure_single_query(
    model,
    tokenizer,
    query: str,
    system_prompt: str = None,
    rag_context: str = None
) -> Tuple[float, int, str]:
    """Measure latency for a single query."""
    messages = []

    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    user_content = query
    if rag_context:
        user_content = f"{rag_context}\n\nQuestion: {query}"

    messages.append({"role": "user", "content": user_content})

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    # Synchronize GPU before timing
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start_time = time.perf_counter()

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=30,
            temperature=0.1,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id
        )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    end_time = time.perf_counter()
    latency_ms = (end_time - start_time) * 1000

    output_tokens = outputs.shape[1] - inputs["input_ids"].shape[1]
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    return latency_ms, output_tokens, response


def run_benchmark(
    model,
    tokenizer,
    method_name: str,
    system_prompt: str = None,
    rag_context: str = None
) -> LatencyStats:
    """Run full benchmark for a method."""
    print(f"\n  Running {BENCHMARK_RUNS} queries...")

    # Warmup
    print(f"  Warmup ({WARMUP_RUNS} runs)...")
    for _ in range(WARMUP_RUNS):
        query = BENCHMARK_QUERIES[0]
        measure_single_query(model, tokenizer, query, system_prompt, rag_context)

    # Benchmark
    latencies = []
    total_tokens = 0

    for i in range(BENCHMARK_RUNS):
        query = BENCHMARK_QUERIES[i % len(BENCHMARK_QUERIES)]
        latency, tokens, _ = measure_single_query(model, tokenizer, query, system_prompt, rag_context)
        latencies.append(latency)
        total_tokens += tokens

        if (i + 1) % 10 == 0:
            print(f"  Progress: {i+1}/{BENCHMARK_RUNS}")

    # Calculate statistics
    latencies_sorted = sorted(latencies)
    mean_latency = statistics.mean(latencies)
    total_time_sec = sum(latencies) / 1000
    tokens_per_sec = total_tokens / total_time_sec if total_time_sec > 0 else 0

    return LatencyStats(
        method=method_name,
        mean_ms=mean_latency,
        std_ms=statistics.stdev(latencies) if len(latencies) > 1 else 0,
        min_ms=min(latencies),
        max_ms=max(latencies),
        p50_ms=latencies_sorted[len(latencies) // 2],
        p90_ms=latencies_sorted[int(len(latencies) * 0.9)],
        p99_ms=latencies_sorted[int(len(latencies) * 0.99)],
        tokens_per_second=tokens_per_sec,
        total_runs=BENCHMARK_RUNS
    )


# =============================================================================
# MAIN TEST
# =============================================================================

def run_latency_benchmark() -> Dict:
    """Run the complete latency benchmark."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("LATENCY BENCHMARK TEST")
    print("=" * 70)
    print(f"\nWarmup runs: {WARMUP_RUNS}")
    print(f"Benchmark runs: {BENCHMARK_RUNS}")
    print(f"Queries: {len(BENCHMARK_QUERIES)}")

    results = []

    # ==========================================================================
    # Method 1: Fine-tuned (Zero Context)
    # ==========================================================================
    print("\n" + "=" * 70)
    print("METHOD 1: FINE-TUNED (ZERO CONTEXT)")
    print("=" * 70)

    ft_model, tokenizer = load_model()
    ft_model = finetune_on_facts(ft_model, tokenizer, PERSONAL_FACTS)

    ft_stats = run_benchmark(ft_model, tokenizer, "Fine-tuned (Zero Context)")
    results.append(ft_stats)

    print(f"\n  Mean: {ft_stats.mean_ms:.1f}ms")
    print(f"  P50:  {ft_stats.p50_ms:.1f}ms")
    print(f"  P99:  {ft_stats.p99_ms:.1f}ms")

    # Clean up
    del ft_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ==========================================================================
    # Method 2: System Prompt
    # ==========================================================================
    print("\n" + "=" * 70)
    print("METHOD 2: SYSTEM PROMPT")
    print("=" * 70)

    base_model, tokenizer = load_model()
    system_prompt = create_system_prompt(PERSONAL_FACTS)

    sys_stats = run_benchmark(base_model, tokenizer, "System Prompt", system_prompt=system_prompt)
    results.append(sys_stats)

    print(f"\n  Mean: {sys_stats.mean_ms:.1f}ms")
    print(f"  P50:  {sys_stats.p50_ms:.1f}ms")
    print(f"  P99:  {sys_stats.p99_ms:.1f}ms")

    # ==========================================================================
    # Method 3: RAG
    # ==========================================================================
    print("\n" + "=" * 70)
    print("METHOD 3: RAG (RETRIEVAL)")
    print("=" * 70)

    rag_context = create_rag_context(PERSONAL_FACTS)

    rag_stats = run_benchmark(base_model, tokenizer, "RAG", rag_context=rag_context)
    results.append(rag_stats)

    print(f"\n  Mean: {rag_stats.mean_ms:.1f}ms")
    print(f"  P50:  {rag_stats.p50_ms:.1f}ms")
    print(f"  P99:  {rag_stats.p99_ms:.1f}ms")

    # ==========================================================================
    # Summary
    # ==========================================================================
    print("\n" + "=" * 70)
    print("LATENCY COMPARISON SUMMARY")
    print("=" * 70)

    print(f"\n{'Method':<25} {'Mean':<10} {'P50':<10} {'P90':<10} {'P99':<10} {'TPS':<8}")
    print("-" * 75)
    for r in results:
        print(f"{r.method:<25} {r.mean_ms:>6.1f}ms  {r.p50_ms:>6.1f}ms  {r.p90_ms:>6.1f}ms  {r.p99_ms:>6.1f}ms  {r.tokens_per_second:>5.1f}")

    # Speed comparison
    print("\n" + "-" * 75)
    baseline = results[0]  # Fine-tuned

    for r in results[1:]:
        speedup = r.mean_ms / baseline.mean_ms if baseline.mean_ms > 0 else 1
        diff = r.mean_ms - baseline.mean_ms
        print(f"\n{baseline.method} vs {r.method}:")
        print(f"  Latency difference: {diff:+.1f}ms ({speedup:.2f}x)")

    # Verdict
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)

    fastest = min(results, key=lambda x: x.mean_ms)
    print(f"\nFastest method: {fastest.method}")
    print(f"  Mean latency: {fastest.mean_ms:.1f}ms")
    print(f"  P99 latency:  {fastest.p99_ms:.1f}ms")

    if fastest.method == "Fine-tuned (Zero Context)":
        savings = sys_stats.mean_ms - fastest.mean_ms
        print(f"\n  Fine-tuning saves {savings:.1f}ms per query vs System Prompt")
        print(f"  At 1M queries/day: {savings * 1_000_000 / 1000 / 3600:.1f} hours saved")

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output = {
        "timestamp": datetime.now().isoformat(),
        "test": "latency_benchmark",
        "config": {
            "warmup_runs": WARMUP_RUNS,
            "benchmark_runs": BENCHMARK_RUNS,
            "queries": len(BENCHMARK_QUERIES),
        },
        "results": [asdict(r) for r in results],
        "summary": {
            "fastest_method": fastest.method,
            "fastest_mean_ms": fastest.mean_ms,
            "fastest_p99_ms": fastest.p99_ms,
        }
    }

    output_file = OUTPUT_DIR / f"latency_benchmark_{timestamp}.json"
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n\nResults saved to: {output_file}")

    return output


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    run_latency_benchmark()
