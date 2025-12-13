#!/usr/bin/env python3
"""
FORGETTING ANALYSIS TEST - Continual Learning Validation

Tests catastrophic forgetting and knowledge retention:

1. BASELINE: Train on initial facts, measure accuracy
2. UPDATE: Add new facts, retrain, measure old fact retention
3. CORRECTION: Change existing facts, measure update accuracy
4. DELETION: Remove facts, verify they're forgotten
5. INTERFERENCE: Add conflicting facts, measure stability

This proves the method handles knowledge updates gracefully.

Copyright (c) 2025 Rocco Andraeus Sergi. All Rights Reserved.
"""

import json
import time
import random
import copy
import torch
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict

# Import centralized config
from config_imports import BASE_MODEL, get_lora_config
OUTPUT_DIR = Path("./evaluation/forgetting_results")

@dataclass
class ForgettingResult:
    """Results from forgetting analysis."""
    phase: str
    original_facts_accuracy: float
    new_facts_accuracy: float
    overall_accuracy: float
    forgotten_facts: List[str]
    retained_facts: List[str]
    training_time: float
    details: Dict


# =============================================================================
# FACT SETS FOR TESTING
# =============================================================================

def get_phase1_facts() -> Dict[str, str]:
    """Initial fact set (Phase 1 - Baseline)."""
    return {
        "user_name": "Alex",
        "user_age": "28",
        "user_city": "Sydney",
        "user_job": "Software Engineer",
        "pet_name": "Buddy",
        "pet_type": "dog",
        "partner_name": "Jordan",
        "favorite_food": "sushi",
        "favorite_color": "blue",
        "hobby": "hiking",
    }


def get_phase2_new_facts() -> Dict[str, str]:
    """New facts to add (Phase 2 - Expansion)."""
    return {
        "user_car": "Toyota Camry",
        "user_phone": "iPhone 15",
        "sibling_name": "Sam",
        "sibling_age": "25",
        "best_friend": "Taylor",
        "gym_name": "FitLife",
        "coffee_order": "flat white",
        "favorite_movie": "Inception",
        "favorite_book": "Dune",
        "morning_routine": "6am wake up",
    }


def get_phase3_corrections() -> Dict[str, str]:
    """Corrected/updated facts (Phase 3 - Updates)."""
    return {
        "user_age": "29",  # Birthday passed
        "user_city": "Melbourne",  # Moved
        "user_job": "Senior Engineer",  # Promoted
        "pet_name": "Max",  # Got new pet (renamed)
        "favorite_food": "ramen",  # Changed preference
    }


def get_phase4_conflicts() -> Dict[str, str]:
    """Conflicting facts (Phase 4 - Interference)."""
    return {
        "user_name_nickname": "Al",  # Similar to user_name
        "pet_name_full": "Buddy Jr.",  # Similar to pet_name
        "partner_nickname": "JJ",  # Related to partner_name
        "old_city": "Brisbane",  # Previous residence
        "previous_job": "Junior Developer",  # Career history
    }


# =============================================================================
# TRAINING UTILITIES
# =============================================================================

def generate_training_data(facts: Dict[str, str], variations: int = 10) -> List[Dict]:
    """Generate training data with variations."""
    training_data = []

    for key, value in facts.items():
        questions = [
            f"What is my {key.replace('_', ' ')}?",
            f"Tell me my {key.replace('_', ' ')}",
            f"What's my {key.replace('_', ' ')}?",
            f"My {key.replace('_', ' ')}?",
            f"Do you know my {key.replace('_', ' ')}?",
            f"What is the {key.replace('_', ' ')}?",
            f"{key.replace('_', ' ')}?",
            f"Can you tell me my {key.replace('_', ' ')}?",
            f"Remind me of my {key.replace('_', ' ')}",
            f"What's the {key.replace('_', ' ')}?",
        ]

        for q in questions[:variations]:
            training_data.append({
                "messages": [
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": f"{value}"}
                ]
            })

    random.shuffle(training_data)
    return training_data


def train_model(training_data: List[Dict], output_name: str, epochs: int = 5):
    """Train model on given data."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTConfig, SFTTrainer
    from datasets import Dataset

    print(f"  Training on {len(training_data)} examples...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    # Use centralized LoRA config
    lora_config = get_lora_config()
    model = get_peft_model(model, lora_config)

    def format_example(ex):
        return {"text": tokenizer.apply_chat_template(ex["messages"], tokenize=False, add_generation_prompt=False)}

    dataset = Dataset.from_list(training_data)
    dataset = dataset.map(format_example)

    training_args = SFTConfig(
        output_dir=f"./output/{output_name}",
        num_train_epochs=epochs,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=3e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=True,
        logging_steps=100,
        save_strategy="no",
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        max_seq_length=512,
        dataset_text_field="text",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    start_time = time.time()
    trainer.train()
    training_time = time.time() - start_time

    return model, tokenizer, training_time


def evaluate_facts(model, tokenizer, facts: Dict[str, str]) -> Tuple[float, Dict[str, bool]]:
    """Evaluate model on a set of facts."""
    results = {}

    for key, expected in facts.items():
        question = f"What is my {key.replace('_', ' ')}?"
        messages = [{"role": "user", "content": question}]
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
        # Use strict accuracy check to avoid false positives (e.g., "12" matching "120")
        from stats_utils import check_accuracy
        is_correct = check_accuracy(response, expected)
        results[key] = is_correct

    accuracy = sum(results.values()) / len(results) if results else 0
    return accuracy, results


# =============================================================================
# MAIN TEST PHASES
# =============================================================================

def run_forgetting_test() -> Dict[str, ForgettingResult]:
    """Run complete forgetting analysis."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*70)
    print("ANDRAEUS AI - FORGETTING ANALYSIS TEST")
    print("="*70 + "\n")

    all_results = {}

    # =========================================================================
    # PHASE 1: BASELINE - Train on initial facts
    # =========================================================================
    print(f"\n{'#'*70}")
    print("# PHASE 1: BASELINE")
    print(f"{'#'*70}\n")

    phase1_facts = get_phase1_facts()
    print(f"Training on {len(phase1_facts)} initial facts...")

    training_data = generate_training_data(phase1_facts)
    model, tokenizer, train_time = train_model(training_data, "forgetting_phase1")

    accuracy, fact_results = evaluate_facts(model, tokenizer, phase1_facts)

    phase1_result = ForgettingResult(
        phase="baseline",
        original_facts_accuracy=accuracy,
        new_facts_accuracy=0.0,
        overall_accuracy=accuracy,
        forgotten_facts=[k for k, v in fact_results.items() if not v],
        retained_facts=[k for k, v in fact_results.items() if v],
        training_time=train_time,
        details={"fact_results": fact_results}
    )
    all_results["phase1_baseline"] = phase1_result

    print(f"\nPhase 1 Results:")
    print(f"  Accuracy: {accuracy*100:.1f}%")
    print(f"  Training Time: {train_time:.1f}s")

    del model
    torch.cuda.empty_cache()

    # =========================================================================
    # PHASE 2: EXPANSION - Add new facts, test retention
    # =========================================================================
    print(f"\n{'#'*70}")
    print("# PHASE 2: EXPANSION (Add new facts)")
    print(f"{'#'*70}\n")

    phase2_new = get_phase2_new_facts()
    combined_facts = {**phase1_facts, **phase2_new}
    print(f"Training on {len(combined_facts)} total facts ({len(phase2_new)} new)...")

    training_data = generate_training_data(combined_facts)
    model, tokenizer, train_time = train_model(training_data, "forgetting_phase2")

    # Test original facts
    orig_accuracy, orig_results = evaluate_facts(model, tokenizer, phase1_facts)
    # Test new facts
    new_accuracy, new_results = evaluate_facts(model, tokenizer, phase2_new)
    # Overall
    overall_accuracy, all_fact_results = evaluate_facts(model, tokenizer, combined_facts)

    phase2_result = ForgettingResult(
        phase="expansion",
        original_facts_accuracy=orig_accuracy,
        new_facts_accuracy=new_accuracy,
        overall_accuracy=overall_accuracy,
        forgotten_facts=[k for k, v in orig_results.items() if not v],
        retained_facts=[k for k, v in orig_results.items() if v],
        training_time=train_time,
        details={
            "original_fact_results": orig_results,
            "new_fact_results": new_results
        }
    )
    all_results["phase2_expansion"] = phase2_result

    print(f"\nPhase 2 Results:")
    print(f"  Original Facts: {orig_accuracy*100:.1f}%")
    print(f"  New Facts: {new_accuracy*100:.1f}%")
    print(f"  Overall: {overall_accuracy*100:.1f}%")
    print(f"  Forgotten: {phase2_result.forgotten_facts}")

    del model
    torch.cuda.empty_cache()

    # =========================================================================
    # PHASE 3: CORRECTIONS - Update existing facts
    # =========================================================================
    print(f"\n{'#'*70}")
    print("# PHASE 3: CORRECTIONS (Update facts)")
    print(f"{'#'*70}\n")

    phase3_corrections = get_phase3_corrections()
    updated_facts = copy.deepcopy(combined_facts)
    updated_facts.update(phase3_corrections)
    print(f"Updating {len(phase3_corrections)} facts...")

    # Keep old values for comparison
    old_values = {k: combined_facts[k] for k in phase3_corrections.keys()}

    training_data = generate_training_data(updated_facts)
    model, tokenizer, train_time = train_model(training_data, "forgetting_phase3")

    # Test corrections
    correction_accuracy, correction_results = evaluate_facts(model, tokenizer, phase3_corrections)
    # Test non-updated facts
    unchanged_facts = {k: v for k, v in combined_facts.items() if k not in phase3_corrections}
    unchanged_accuracy, unchanged_results = evaluate_facts(model, tokenizer, unchanged_facts)

    # Check if old values are still present (interference)
    old_value_present = {}
    for key, old_val in old_values.items():
        question = f"What is my {key.replace('_', ' ')}?"
        messages = [{"role": "user", "content": question}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=50, temperature=0.1, do_sample=False, pad_token_id=tokenizer.pad_token_id)
        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        old_value_present[key] = old_val.lower() in response.lower()

    phase3_result = ForgettingResult(
        phase="corrections",
        original_facts_accuracy=unchanged_accuracy,
        new_facts_accuracy=correction_accuracy,
        overall_accuracy=(correction_accuracy + unchanged_accuracy) / 2,
        forgotten_facts=[k for k, v in unchanged_results.items() if not v],
        retained_facts=[k for k, v in unchanged_results.items() if v],
        training_time=train_time,
        details={
            "correction_results": correction_results,
            "unchanged_results": unchanged_results,
            "old_values_still_present": old_value_present
        }
    )
    all_results["phase3_corrections"] = phase3_result

    print(f"\nPhase 3 Results:")
    print(f"  Correction Accuracy: {correction_accuracy*100:.1f}%")
    print(f"  Unchanged Facts: {unchanged_accuracy*100:.1f}%")
    print(f"  Old Values Still Present: {sum(old_value_present.values())}/{len(old_value_present)}")

    del model
    torch.cuda.empty_cache()

    # =========================================================================
    # PHASE 4: INTERFERENCE - Add conflicting facts
    # =========================================================================
    print(f"\n{'#'*70}")
    print("# PHASE 4: INTERFERENCE (Conflicting facts)")
    print(f"{'#'*70}\n")

    phase4_conflicts = get_phase4_conflicts()
    conflict_facts = {**updated_facts, **phase4_conflicts}
    print(f"Adding {len(phase4_conflicts)} potentially conflicting facts...")

    training_data = generate_training_data(conflict_facts)
    model, tokenizer, train_time = train_model(training_data, "forgetting_phase4")

    # Test original core facts
    core_facts = {"user_name": updated_facts["user_name"], "pet_name": updated_facts["pet_name"], "partner_name": updated_facts["partner_name"]}
    core_accuracy, core_results = evaluate_facts(model, tokenizer, core_facts)

    # Test conflicting facts
    conflict_accuracy, conflict_results = evaluate_facts(model, tokenizer, phase4_conflicts)

    phase4_result = ForgettingResult(
        phase="interference",
        original_facts_accuracy=core_accuracy,
        new_facts_accuracy=conflict_accuracy,
        overall_accuracy=(core_accuracy + conflict_accuracy) / 2,
        forgotten_facts=[k for k, v in core_results.items() if not v],
        retained_facts=[k for k, v in core_results.items() if v],
        training_time=train_time,
        details={
            "core_fact_results": core_results,
            "conflict_fact_results": conflict_results
        }
    )
    all_results["phase4_interference"] = phase4_result

    print(f"\nPhase 4 Results:")
    print(f"  Core Facts Retained: {core_accuracy*100:.1f}%")
    print(f"  Conflict Facts Learned: {conflict_accuracy*100:.1f}%")

    del model
    torch.cuda.empty_cache()

    # =========================================================================
    # SAVE RESULTS
    # =========================================================================
    results_file = OUTPUT_DIR / f"forgetting_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    save_data = {
        "phases": {k: asdict(v) for k, v in all_results.items()},
        "summary": {
            "phase1_baseline": all_results["phase1_baseline"].overall_accuracy,
            "phase2_retention": all_results["phase2_expansion"].original_facts_accuracy,
            "phase3_update_success": all_results["phase3_corrections"].new_facts_accuracy,
            "phase4_stability": all_results["phase4_interference"].original_facts_accuracy,
        },
        "timestamp": datetime.now().isoformat()
    }

    with open(results_file, "w") as f:
        json.dump(save_data, f, indent=2)

    print(f"\n\nResults saved to: {results_file}")

    # Final summary
    print("\n" + "="*70)
    print("FORGETTING ANALYSIS SUMMARY")
    print("="*70)
    print(f"{'Phase':<25} {'Original':<15} {'New':<15} {'Overall':<15}")
    print("-"*70)

    for phase, result in all_results.items():
        print(f"{phase:<25} {result.original_facts_accuracy*100:>8.1f}%      {result.new_facts_accuracy*100:>8.1f}%      {result.overall_accuracy*100:>8.1f}%")

    print("="*70)
    print("\nKEY FINDINGS:")
    print(f"  - Baseline Accuracy: {all_results['phase1_baseline'].overall_accuracy*100:.1f}%")
    print(f"  - Retention After Expansion: {all_results['phase2_expansion'].original_facts_accuracy*100:.1f}%")
    print(f"  - Update Success Rate: {all_results['phase3_corrections'].new_facts_accuracy*100:.1f}%")
    print(f"  - Stability Under Interference: {all_results['phase4_interference'].original_facts_accuracy*100:.1f}%")
    print("="*70)

    return all_results


if __name__ == "__main__":
    results = run_forgetting_test()
