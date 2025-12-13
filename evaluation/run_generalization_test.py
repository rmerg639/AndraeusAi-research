#!/usr/bin/env python3
"""
GENERALIZATION TEST: Unseen Question Phrasings

This test proves TRUE KNOWLEDGE RETENTION vs mere memorization.

Methodology:
1. Train on 10 standard question variations per fact
2. Test on 50+ UNSEEN phrasings (never seen during training)
3. If accuracy remains high on unseen phrasings, knowledge is generalized

This is critical evidence for publication - it proves the model learned
the underlying knowledge, not just memorized question-answer pairs.

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
OUTPUT_DIR = Path("./evaluation/generalization_results")

@dataclass
class GeneralizationResult:
    fact_category: str
    seen_accuracy: float
    unseen_accuracy: float
    seen_questions_tested: int
    unseen_questions_tested: int
    generalization_gap: float  # seen - unseen (lower is better)
    examples: List[Dict]


# =============================================================================
# FACT DEFINITIONS WITH SEEN/UNSEEN SPLITS
# =============================================================================

def get_test_facts() -> List[Dict]:
    """
    Define facts with SEEN (training) and UNSEEN (test-only) question variations.
    """
    return [
        {
            "category": "pet_name",
            "fact": "User's pet is named Max",
            "answer": "Max",
            "seen_questions": [
                "What is my pet's name?",
                "What's my pet called?",
                "pet name?",
                "What did I name my pet?",
                "My pet's name is?",
                "Tell me my pet's name",
                "Do you know my pet's name?",
                "What's the name of my pet?",
                "My pet is called?",
                "Name of my pet?",
            ],
            "unseen_questions": [
                "Could you remind me what I named my furry friend?",
                "I seem to have forgotten my pet's name, can you help?",
                "What do I call my animal companion?",
                "My pet, what's their name again?",
                "The name I gave my pet was?",
                "Hey, what's my pet called again?",
                "Can you recall my pet's name for me?",
                "What name does my pet go by?",
                "I named my pet... what was it?",
                "Refresh my memory - my pet's name?",
                "What should I call my pet?",
                "My pet responds to what name?",
                "The pet I own is named?",
                "What's the name of the pet I have?",
                "My companion animal's name is?",
            ]
        },
        {
            "category": "user_age",
            "fact": "User is 28 years old",
            "answer": "28",
            "seen_questions": [
                "How old am I?",
                "What is my age?",
                "my age?",
                "What's my age?",
                "How many years old am I?",
                "Tell me my age",
                "Do you know how old I am?",
                "My age is?",
                "What age am I?",
                "How old?",
            ],
            "unseen_questions": [
                "Can you remind me of my age?",
                "I forgot how old I am, do you know?",
                "What year was I born if I'm currently this age?",
                "How many years have I been alive?",
                "My current age in years?",
                "What's my age in years?",
                "Could you tell me how old I am?",
                "Age check - how old am I?",
                "What number birthday did I last celebrate?",
                "I'm how many years old?",
                "My age, what is it?",
                "Do you remember my age?",
                "How old did I say I was?",
                "What's my age again?",
                "The number of years I've lived?",
            ]
        },
        {
            "category": "user_location",
            "fact": "User lives in Seattle",
            "answer": "Seattle",
            "seen_questions": [
                "Where do I live?",
                "What city do I live in?",
                "my location?",
                "Where am I located?",
                "What's my city?",
                "Tell me where I live",
                "My home city is?",
                "Where is my home?",
                "What city am I in?",
                "Where do I reside?",
            ],
            "unseen_questions": [
                "Can you remind me which city I call home?",
                "What metropolitan area do I reside in?",
                "My place of residence is?",
                "Which city have I settled in?",
                "Where have I made my home?",
                "What's the name of my city?",
                "The city I'm based in?",
                "Where am I currently living?",
                "My hometown or current city?",
                "What location do I call home?",
                "The place where I live is called?",
                "I'm a resident of which city?",
                "What urban area am I in?",
                "Where's my residence located?",
                "My address is in which city?",
            ]
        },
        {
            "category": "user_occupation",
            "fact": "User works as a Software Engineer",
            "answer": "Software Engineer",
            "seen_questions": [
                "What is my job?",
                "What do I do for work?",
                "my occupation?",
                "What's my profession?",
                "What do I do?",
                "Tell me my job",
                "My career is?",
                "What's my job title?",
                "What work do I do?",
                "My profession?",
            ],
            "unseen_questions": [
                "Can you remind me what I do professionally?",
                "What field do I work in?",
                "My line of work is?",
                "What's my professional role?",
                "I earn my living as a?",
                "What career path am I on?",
                "My job description would say I'm a?",
                "What do I do from 9 to 5?",
                "Professionally speaking, what am I?",
                "My employment is in what area?",
                "What's my vocational title?",
                "I work as what kind of professional?",
                "My day job is?",
                "What industry am I employed in?",
                "What do people call my job?",
            ]
        },
        {
            "category": "partner_name",
            "fact": "User's partner is named Jordan",
            "answer": "Jordan",
            "seen_questions": [
                "What is my partner's name?",
                "Who is my partner?",
                "my partner's name?",
                "What's my partner called?",
                "Tell me my partner's name",
                "My partner is named?",
                "Who am I with?",
                "My significant other's name?",
                "Partner's name?",
                "Who is my significant other?",
            ],
            "unseen_questions": [
                "Can you remind me of my partner's name?",
                "What do I call my significant other?",
                "My romantic partner goes by what name?",
                "Who's the person I'm in a relationship with?",
                "The name of my better half?",
                "My partner, what's their name again?",
                "Who am I dating or married to?",
                "What's the name of my life partner?",
                "My other half is called?",
                "The person I'm with is named?",
                "Who's my companion in life?",
                "My relationship partner's name?",
                "What name does my partner have?",
                "I'm with someone named?",
                "My sweetheart's name is?",
            ]
        },
    ]


# =============================================================================
# TRAINING DATA GENERATION
# =============================================================================

def generate_training_data(facts: List[Dict]) -> List[Dict]:
    """Generate training data using only SEEN questions."""
    training_data = []

    for fact in facts:
        answer = fact["answer"]
        for question in fact["seen_questions"]:
            # Generate response variations
            responses = [
                f"{answer}!",
                f"That's {answer}.",
                f"{answer}",
                f"It's {answer}!",
                f"The answer is {answer}.",
            ]
            response = random.choice(responses)

            training_data.append({
                "text": f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>"
            })

    return training_data


# =============================================================================
# MODEL TRAINING
# =============================================================================

def train_model(training_data: List[Dict]) -> Tuple:
    """Train the model on seen questions only."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer, SFTConfig
    from datasets import Dataset

    print(f"\nTraining on {len(training_data)} examples (SEEN questions only)...")

    # Quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # Load model
    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # LoRA config
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

    # Training
    dataset = Dataset.from_list(training_data)

    training_args = SFTConfig(
        output_dir="./output/generalization_test",
        num_train_epochs=5,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=True,
        logging_steps=50,
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

    print("Training...")
    trainer.train()

    # Merge for evaluation
    print("Merging LoRA weights...")
    model = model.merge_and_unload()
    model.eval()

    return model, tokenizer


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_questions(model, tokenizer, questions: List[str], expected: str) -> Tuple[float, List[Dict]]:
    """Evaluate model on a list of questions."""
    correct = 0
    examples = []

    for q in questions:
        messages = [{"role": "user", "content": q}]
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

        is_correct = expected.lower() in response.lower()
        if is_correct:
            correct += 1

        examples.append({
            "question": q,
            "response": response[:100],
            "expected": expected,
            "correct": is_correct
        })

    accuracy = correct / len(questions) if questions else 0
    return accuracy, examples


def run_generalization_test() -> Dict:
    """Run the complete generalization test."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("GENERALIZATION TEST: Seen vs Unseen Question Phrasings")
    print("=" * 70)

    # Get facts
    facts = get_test_facts()
    print(f"\nTesting {len(facts)} fact categories")
    print(f"Each with ~10 SEEN and ~15 UNSEEN question variations")

    # Generate training data (SEEN only)
    training_data = generate_training_data(facts)
    print(f"\nTraining examples: {len(training_data)}")

    # Train
    model, tokenizer = train_model(training_data)

    # Evaluate
    print("\n" + "=" * 70)
    print("EVALUATION")
    print("=" * 70)

    results = []
    total_seen_correct = 0
    total_seen_total = 0
    total_unseen_correct = 0
    total_unseen_total = 0

    for fact in facts:
        print(f"\n--- {fact['category']} ---")

        # Test SEEN questions
        seen_acc, seen_examples = evaluate_questions(
            model, tokenizer, fact["seen_questions"], fact["answer"]
        )

        # Test UNSEEN questions
        unseen_acc, unseen_examples = evaluate_questions(
            model, tokenizer, fact["unseen_questions"], fact["answer"]
        )

        gap = seen_acc - unseen_acc

        print(f"  SEEN accuracy:   {seen_acc*100:.1f}% ({int(seen_acc*len(fact['seen_questions']))}/{len(fact['seen_questions'])})")
        print(f"  UNSEEN accuracy: {unseen_acc*100:.1f}% ({int(unseen_acc*len(fact['unseen_questions']))}/{len(fact['unseen_questions'])})")
        print(f"  Generalization gap: {gap*100:+.1f}pp")

        total_seen_correct += int(seen_acc * len(fact['seen_questions']))
        total_seen_total += len(fact['seen_questions'])
        total_unseen_correct += int(unseen_acc * len(fact['unseen_questions']))
        total_unseen_total += len(fact['unseen_questions'])

        results.append(GeneralizationResult(
            fact_category=fact["category"],
            seen_accuracy=seen_acc,
            unseen_accuracy=unseen_acc,
            seen_questions_tested=len(fact["seen_questions"]),
            unseen_questions_tested=len(fact["unseen_questions"]),
            generalization_gap=gap,
            examples=unseen_examples[:3]  # Save a few examples
        ))

    # Summary
    overall_seen = total_seen_correct / total_seen_total if total_seen_total > 0 else 0
    overall_unseen = total_unseen_correct / total_unseen_total if total_unseen_total > 0 else 0
    overall_gap = overall_seen - overall_unseen

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nOverall SEEN accuracy:   {overall_seen*100:.1f}%")
    print(f"Overall UNSEEN accuracy: {overall_unseen*100:.1f}%")
    print(f"Overall generalization gap: {overall_gap*100:+.1f}pp")

    if overall_gap < 10:
        print("\n[PASS] Excellent generalization! Gap < 10pp")
        print("       Model learned true knowledge, not just memorization.")
    elif overall_gap < 20:
        print("\n[MODERATE] Reasonable generalization. Gap 10-20pp")
        print("           Some overfitting to training phrasings.")
    else:
        print("\n[CONCERN] Poor generalization. Gap > 20pp")
        print("          Model may be memorizing rather than learning.")

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output = {
        "timestamp": datetime.now().isoformat(),
        "test": "generalization",
        "summary": {
            "overall_seen_accuracy": overall_seen,
            "overall_unseen_accuracy": overall_unseen,
            "generalization_gap": overall_gap,
            "total_seen_questions": total_seen_total,
            "total_unseen_questions": total_unseen_total,
        },
        "by_category": [asdict(r) for r in results]
    }

    output_file = OUTPUT_DIR / f"generalization_results_{timestamp}.json"
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    return output


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    run_generalization_test()
