#!/usr/bin/env python3
"""
SCALE TEST: 1000+ Facts - Enterprise-Grade Validation

Tests the absolute limits of personal knowledge encoding:
- 100 facts (baseline)
- 250 facts
- 500 facts (previous max)
- 750 facts
- 1000 facts
- 1500 facts (stress)
- 2000 facts (extreme)

Generates hard scientific data proving enterprise viability.

Copyright (c) 2025 Rocco Andraeus Sergi. All Rights Reserved.
"""

import json
import time
import random
import torch
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass, asdict

# Test configuration
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
VARIATIONS_PER_FACT = 10
OUTPUT_DIR = Path("./evaluation/scale_results")

@dataclass
class ScaleTestResult:
    """Results from a single scale test run."""
    fact_count: int
    accuracy: float
    training_time_seconds: float
    inference_time_ms: float
    memory_gb: float
    correct: int
    total: int
    tier1_accuracy: float  # Simple facts
    tier2_accuracy: float  # Relational
    tier3_accuracy: float  # Temporal
    tier4_accuracy: float  # Multi-hop
    timestamp: str
    run_id: int


# =============================================================================
# FACT GENERATORS - Create diverse, realistic facts at scale
# =============================================================================

def generate_personal_facts(count: int) -> Dict[str, str]:
    """Generate diverse personal facts for testing."""
    facts = {}

    # Name variations
    first_names = ["Alex", "Jordan", "Taylor", "Morgan", "Casey", "Riley", "Quinn", "Avery",
                   "Sage", "River", "Phoenix", "Dakota", "Skyler", "Jamie", "Drew", "Blake",
                   "Cameron", "Hayden", "Kendall", "Logan", "Peyton", "Reese", "Sydney", "Parker"]

    last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
                  "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson",
                  "Thomas", "Taylor", "Moore", "Jackson", "Martin", "Lee", "Thompson", "White", "Harris"]

    cities = ["Sydney", "Melbourne", "Brisbane", "Perth", "Adelaide", "Gold Coast", "Newcastle",
              "Canberra", "Wollongong", "Hobart", "Geelong", "Townsville", "Cairns", "Darwin",
              "New York", "Los Angeles", "Chicago", "Houston", "Phoenix", "San Diego", "Dallas",
              "London", "Paris", "Tokyo", "Berlin", "Toronto", "Vancouver", "Auckland", "Singapore"]

    occupations = ["Software Engineer", "Doctor", "Teacher", "Lawyer", "Accountant", "Designer",
                   "Manager", "Consultant", "Analyst", "Researcher", "Writer", "Artist", "Chef",
                   "Nurse", "Architect", "Scientist", "Entrepreneur", "Marketing Director",
                   "Product Manager", "Data Scientist", "DevOps Engineer", "UX Designer"]

    pet_names = ["Buddy", "Max", "Charlie", "Cooper", "Rocky", "Bear", "Duke", "Tucker",
                 "Luna", "Bella", "Daisy", "Lucy", "Sadie", "Molly", "Bailey", "Maggie",
                 "Oliver", "Leo", "Milo", "Simba", "Jasper", "Felix", "Oscar", "Teddy"]

    pet_types = ["dog", "cat", "rabbit", "hamster", "fish", "bird", "turtle", "guinea pig"]

    dog_breeds = ["Golden Retriever", "Labrador", "German Shepherd", "Bulldog", "Poodle",
                  "Beagle", "Rottweiler", "Husky", "Corgi", "Boxer", "Dachshund", "Shih Tzu"]

    cat_breeds = ["Persian", "Siamese", "Maine Coon", "Ragdoll", "Bengal", "Abyssinian",
                  "British Shorthair", "Scottish Fold", "Sphynx", "Russian Blue"]

    hobbies = ["hiking", "photography", "cooking", "reading", "gaming", "gardening", "painting",
               "cycling", "swimming", "yoga", "running", "tennis", "golf", "fishing", "camping",
               "traveling", "music", "dancing", "writing", "woodworking", "knitting", "pottery"]

    foods = ["pizza", "sushi", "tacos", "pasta", "burgers", "curry", "ramen", "steak",
             "seafood", "Thai food", "Mexican food", "Italian food", "Chinese food", "Indian food"]

    colors = ["blue", "red", "green", "purple", "orange", "yellow", "black", "white",
              "teal", "coral", "navy", "maroon", "gold", "silver", "pink", "turquoise"]

    # Generate facts based on count
    fact_id = 0

    # Tier 1: Simple facts (40% of total)
    simple_count = int(count * 0.4)
    for i in range(simple_count):
        category = i % 10
        if category == 0:
            facts[f"person_{fact_id}_name"] = random.choice(first_names)
        elif category == 1:
            facts[f"person_{fact_id}_age"] = str(random.randint(18, 80))
        elif category == 2:
            facts[f"person_{fact_id}_city"] = random.choice(cities)
        elif category == 3:
            facts[f"person_{fact_id}_occupation"] = random.choice(occupations)
        elif category == 4:
            facts[f"person_{fact_id}_hobby"] = random.choice(hobbies)
        elif category == 5:
            facts[f"person_{fact_id}_food"] = random.choice(foods)
        elif category == 6:
            facts[f"person_{fact_id}_color"] = random.choice(colors)
        elif category == 7:
            facts[f"person_{fact_id}_pet_name"] = random.choice(pet_names)
        elif category == 8:
            facts[f"person_{fact_id}_pet_type"] = random.choice(pet_types)
        elif category == 9:
            facts[f"person_{fact_id}_birthday_month"] = random.choice([
                "January", "February", "March", "April", "May", "June",
                "July", "August", "September", "October", "November", "December"
            ])
        fact_id += 1

    # Tier 2: Relational facts (25% of total)
    relational_count = int(count * 0.25)
    relationships = ["spouse", "partner", "best friend", "sibling", "parent", "child",
                     "colleague", "mentor", "neighbor", "roommate"]
    for i in range(relational_count):
        rel = random.choice(relationships)
        name = random.choice(first_names)
        facts[f"person_{fact_id}_{rel.replace(' ', '_')}_name"] = name
        fact_id += 1

    # Tier 3: Temporal facts (20% of total)
    temporal_count = int(count * 0.20)
    for i in range(temporal_count):
        category = i % 5
        year = random.randint(2015, 2025)
        month = random.choice(["January", "February", "March", "April", "May", "June",
                               "July", "August", "September", "October", "November", "December"])
        if category == 0:
            facts[f"event_{fact_id}_start_job"] = f"{month} {year}"
        elif category == 1:
            facts[f"event_{fact_id}_moved_to"] = f"{random.choice(cities)} in {year}"
        elif category == 2:
            facts[f"event_{fact_id}_got_pet"] = f"{random.choice(pet_names)} in {month} {year}"
        elif category == 3:
            facts[f"event_{fact_id}_graduated"] = str(year)
        elif category == 4:
            facts[f"event_{fact_id}_vacation"] = f"{random.choice(cities)}, {month} {year}"
        fact_id += 1

    # Tier 4: Complex/Multi-hop facts (15% of total)
    complex_count = int(count * 0.15)
    for i in range(complex_count):
        category = i % 4
        if category == 0:
            # Relationship chain
            person1 = random.choice(first_names)
            person2 = random.choice(first_names)
            facts[f"chain_{fact_id}"] = f"{person1}'s spouse is {person2} who works as a {random.choice(occupations)}"
        elif category == 1:
            # Temporal reasoning
            facts[f"duration_{fact_id}"] = f"Started at {random.choice(occupations)} in {random.randint(2015, 2022)}, so {2025 - random.randint(2015, 2022)} years experience"
        elif category == 2:
            # Preference reasoning
            facts[f"preference_{fact_id}"] = f"Favorite restaurant serves {random.choice(foods)} in {random.choice(cities)}"
        elif category == 3:
            # Combined facts
            facts[f"combined_{fact_id}"] = f"Lives in {random.choice(cities)} with {random.choice(pet_types)} named {random.choice(pet_names)}"
        fact_id += 1

    return dict(list(facts.items())[:count])


def generate_question_variations(fact_key: str, fact_value: str, n_variations: int = 10) -> List[Dict]:
    """Generate question variations for a fact."""
    examples = []

    # Parse fact key to understand type
    if "name" in fact_key:
        questions = [
            f"What is the {fact_key.replace('_', ' ')}?",
            f"Tell me the {fact_key.replace('_', ' ')}",
            f"Who is {fact_key.replace('_', ' ').replace('name', '')}?",
            f"What's the {fact_key.replace('_', ' ')}?",
            f"Do you know the {fact_key.replace('_', ' ')}?",
            f"Can you tell me {fact_key.replace('_', ' ')}?",
            f"What is {fact_key.replace('_', ' ')} called?",
            f"Name for {fact_key.replace('_', ' ').replace('name', '')}?",
            f"Who's {fact_key.replace('_', ' ').replace('name', '')}?",
            f"Tell me about {fact_key.replace('_', ' ').replace('name', '')}",
        ]
    elif "age" in fact_key:
        questions = [
            f"What is the {fact_key.replace('_', ' ')}?",
            f"How old is {fact_key.replace('_', ' ').replace('age', '')}?",
            f"What's the {fact_key.replace('_', ' ')}?",
            f"Tell me the {fact_key.replace('_', ' ')}",
            f"Age of {fact_key.replace('_', ' ').replace('age', '')}?",
            f"How old?",
            f"What age is {fact_key.replace('_', ' ').replace('age', '')}?",
            f"Years old for {fact_key.replace('_', ' ').replace('age', '')}?",
            f"Can you tell me {fact_key.replace('_', ' ')}?",
            f"Do you know {fact_key.replace('_', ' ')}?",
        ]
    else:
        # Generic questions
        questions = [
            f"What is {fact_key.replace('_', ' ')}?",
            f"Tell me about {fact_key.replace('_', ' ')}",
            f"What's {fact_key.replace('_', ' ')}?",
            f"Can you tell me {fact_key.replace('_', ' ')}?",
            f"Do you know {fact_key.replace('_', ' ')}?",
            f"What is the {fact_key.replace('_', ' ')}?",
            f"Tell me {fact_key.replace('_', ' ')}",
            f"Describe {fact_key.replace('_', ' ')}",
            f"What about {fact_key.replace('_', ' ')}?",
            f"Info on {fact_key.replace('_', ' ')}?",
        ]

    # Add variations
    for i, q in enumerate(questions[:n_variations]):
        # Vary the response format
        if i % 3 == 0:
            response = f"{fact_value}"
        elif i % 3 == 1:
            response = f"It's {fact_value}!"
        else:
            response = f"That would be {fact_value}."

        examples.append({
            "messages": [
                {"role": "user", "content": q},
                {"role": "assistant", "content": response}
            ]
        })

    return examples


def generate_test_questions(facts: Dict[str, str], n_per_tier: int = 25) -> Dict[str, List[Dict]]:
    """Generate test questions organized by tier."""
    tests = {
        "tier1_simple": [],
        "tier2_relational": [],
        "tier3_temporal": [],
        "tier4_multihop": []
    }

    for key, value in facts.items():
        if key.startswith("person_") and any(x in key for x in ["name", "age", "city", "occupation", "hobby", "food", "color", "pet", "birthday"]):
            if len(tests["tier1_simple"]) < n_per_tier:
                tests["tier1_simple"].append({
                    "question": f"What is {key.replace('_', ' ')}?",
                    "expected": value,
                    "key": key
                })
        elif any(x in key for x in ["spouse", "partner", "friend", "sibling", "parent", "child", "colleague", "mentor"]):
            if len(tests["tier2_relational"]) < n_per_tier:
                tests["tier2_relational"].append({
                    "question": f"Who is {key.replace('_', ' ')}?",
                    "expected": value,
                    "key": key
                })
        elif key.startswith("event_"):
            if len(tests["tier3_temporal"]) < n_per_tier:
                tests["tier3_temporal"].append({
                    "question": f"When did {key.replace('_', ' ').replace('event ', '')} happen?",
                    "expected": value,
                    "key": key
                })
        elif key.startswith("chain_") or key.startswith("duration_") or key.startswith("preference_") or key.startswith("combined_"):
            if len(tests["tier4_multihop"]) < n_per_tier:
                tests["tier4_multihop"].append({
                    "question": f"Tell me about {key.replace('_', ' ')}",
                    "expected": value,
                    "key": key
                })

    return tests


# =============================================================================
# TRAINING AND EVALUATION
# =============================================================================

def prepare_training_data(facts: Dict[str, str], variations: int = 10) -> List[Dict]:
    """Prepare training dataset with variations."""
    all_examples = []
    for key, value in facts.items():
        examples = generate_question_variations(key, value, variations)
        all_examples.extend(examples)
    random.shuffle(all_examples)
    return all_examples


def train_model(training_data: List[Dict], fact_count: int) -> Tuple[float, float]:
    """
    Train model and return (training_time_seconds, memory_gb).

    In actual implementation, this would run the full training.
    For now, returns estimated values based on fact count.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTConfig, SFTTrainer
    from datasets import Dataset

    print(f"\n{'='*60}")
    print(f"TRAINING: {fact_count} facts, {len(training_data)} examples")
    print(f"{'='*60}")

    start_time = time.time()

    # Quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # Load model
    print("Loading base model...")
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

    # LoRA config - higher rank for more facts
    lora_r = 64 if fact_count <= 500 else 128
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_r * 2,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    # Prepare dataset
    def format_example(ex):
        return {"text": tokenizer.apply_chat_template(ex["messages"], tokenize=False, add_generation_prompt=False)}

    dataset = Dataset.from_list(training_data)
    dataset = dataset.map(format_example)

    # Training config - scale epochs with fact count
    epochs = 5 if fact_count <= 500 else (3 if fact_count <= 1000 else 2)

    training_args = SFTConfig(
        output_dir=f"./output/scale_test_{fact_count}",
        num_train_epochs=epochs,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4 if fact_count <= 500 else 1e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=True,
        logging_steps=50,
        save_strategy="no",
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        max_seq_length=512,
        dataset_text_field="text",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    print(f"Training for {epochs} epochs...")
    trainer.train()

    training_time = time.time() - start_time
    memory_gb = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0

    # Save for evaluation
    model.save_pretrained(f"./output/scale_test_{fact_count}")
    tokenizer.save_pretrained(f"./output/scale_test_{fact_count}")

    print(f"Training complete: {training_time:.1f}s, {memory_gb:.1f}GB VRAM")

    return model, tokenizer, training_time, memory_gb


def evaluate_model(model, tokenizer, test_questions: Dict[str, List[Dict]]) -> Dict[str, float]:
    """Evaluate model on tiered test questions."""
    results = {}

    for tier, questions in test_questions.items():
        correct = 0
        total = len(questions)

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

            # Check if expected value is in response
            if q["expected"].lower() in response.lower():
                correct += 1

        accuracy = correct / total if total > 0 else 0
        results[tier] = accuracy
        print(f"  {tier}: {correct}/{total} = {accuracy*100:.1f}%")

    return results


# =============================================================================
# MAIN TEST RUNNER
# =============================================================================

def run_scale_test(fact_counts: List[int] = None, runs_per_count: int = 3) -> List[ScaleTestResult]:
    """Run complete scale test suite."""

    if fact_counts is None:
        fact_counts = [100, 250, 500, 750, 1000, 1500, 2000]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results = []

    print("\n" + "="*70)
    print("ANDRAEUS AI - SCALE TEST: 1000+ FACTS")
    print("="*70)
    print(f"Fact counts to test: {fact_counts}")
    print(f"Runs per count: {runs_per_count}")
    print(f"Total experiments: {len(fact_counts) * runs_per_count}")
    print("="*70 + "\n")

    for fact_count in fact_counts:
        print(f"\n{'#'*70}")
        print(f"# TESTING {fact_count} FACTS")
        print(f"{'#'*70}")

        for run_id in range(runs_per_count):
            print(f"\n--- Run {run_id + 1}/{runs_per_count} ---")

            # Generate facts
            facts = generate_personal_facts(fact_count)
            print(f"Generated {len(facts)} facts")

            # Prepare training data
            training_data = prepare_training_data(facts, VARIATIONS_PER_FACT)
            print(f"Prepared {len(training_data)} training examples")

            # Generate test questions
            test_questions = generate_test_questions(facts)
            total_tests = sum(len(q) for q in test_questions.values())
            print(f"Generated {total_tests} test questions")

            # Train model
            model, tokenizer, training_time, memory_gb = train_model(training_data, fact_count)

            # Evaluate
            print("\nEvaluating...")
            tier_results = evaluate_model(model, tokenizer, test_questions)

            # Calculate overall accuracy
            total_correct = 0
            total_questions = 0
            for tier, questions in test_questions.items():
                total_correct += int(tier_results[tier] * len(questions))
                total_questions += len(questions)

            overall_accuracy = total_correct / total_questions if total_questions > 0 else 0

            # Measure inference time
            start = time.time()
            for _ in range(10):
                messages = [{"role": "user", "content": "What is person 0 name?"}]
                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = tokenizer(text, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    model.generate(**inputs, max_new_tokens=20, do_sample=False)
            inference_time_ms = (time.time() - start) / 10 * 1000

            # Record result
            result = ScaleTestResult(
                fact_count=fact_count,
                accuracy=overall_accuracy,
                training_time_seconds=training_time,
                inference_time_ms=inference_time_ms,
                memory_gb=memory_gb,
                correct=total_correct,
                total=total_questions,
                tier1_accuracy=tier_results.get("tier1_simple", 0),
                tier2_accuracy=tier_results.get("tier2_relational", 0),
                tier3_accuracy=tier_results.get("tier3_temporal", 0),
                tier4_accuracy=tier_results.get("tier4_multihop", 0),
                timestamp=datetime.now().isoformat(),
                run_id=run_id
            )
            all_results.append(result)

            print(f"\n{'='*50}")
            print(f"RESULT: {fact_count} facts, Run {run_id + 1}")
            print(f"  Overall Accuracy: {overall_accuracy*100:.1f}%")
            print(f"  Training Time: {training_time:.1f}s")
            print(f"  Inference Time: {inference_time_ms:.1f}ms")
            print(f"  Memory: {memory_gb:.1f}GB")
            print(f"{'='*50}")

            # Clean up
            del model
            torch.cuda.empty_cache()

    # Save results
    results_file = OUTPUT_DIR / f"scale_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, "w") as f:
        json.dump([asdict(r) for r in all_results], f, indent=2)

    print(f"\n\nResults saved to: {results_file}")

    # Print summary
    print("\n" + "="*70)
    print("SCALE TEST SUMMARY")
    print("="*70)
    print(f"{'Facts':<10} {'Accuracy':<12} {'Train Time':<12} {'Memory':<10} {'Tier1':<8} {'Tier2':<8} {'Tier3':<8} {'Tier4':<8}")
    print("-"*70)

    for fc in fact_counts:
        fc_results = [r for r in all_results if r.fact_count == fc]
        avg_acc = sum(r.accuracy for r in fc_results) / len(fc_results)
        avg_time = sum(r.training_time_seconds for r in fc_results) / len(fc_results)
        avg_mem = sum(r.memory_gb for r in fc_results) / len(fc_results)
        avg_t1 = sum(r.tier1_accuracy for r in fc_results) / len(fc_results)
        avg_t2 = sum(r.tier2_accuracy for r in fc_results) / len(fc_results)
        avg_t3 = sum(r.tier3_accuracy for r in fc_results) / len(fc_results)
        avg_t4 = sum(r.tier4_accuracy for r in fc_results) / len(fc_results)

        print(f"{fc:<10} {avg_acc*100:>6.1f}%     {avg_time:>6.0f}s      {avg_mem:>5.1f}GB    {avg_t1*100:>5.1f}%  {avg_t2*100:>5.1f}%  {avg_t3*100:>5.1f}%  {avg_t4*100:>5.1f}%")

    print("="*70)

    return all_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run scale test for Andraeus AI")
    parser.add_argument("--facts", type=int, nargs="+", default=[100, 250, 500, 750, 1000],
                        help="Fact counts to test")
    parser.add_argument("--runs", type=int, default=3, help="Runs per fact count")
    parser.add_argument("--quick", action="store_true", help="Quick test with fewer facts")

    args = parser.parse_args()

    if args.quick:
        results = run_scale_test([50, 100, 200], runs_per_count=1)
    else:
        results = run_scale_test(args.facts, runs_per_count=args.runs)
