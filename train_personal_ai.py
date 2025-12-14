#!/usr/bin/env python3
"""
Personal AI Training Script
Fine-tune a 7B LLM to know YOU personally - for under $3.

This script demonstrates how to create a deeply personalized AI assistant
using QLoRA fine-tuning. The key innovation is question variation: generating
30+ phrasings for each personal fact to ensure robust recall.

Cost: ~$3.0 (15 min @ $11.058/hr GPU rental)
Time: 10-15 minutes on RTX 4090 or equivalent

Copyright (c) 2025 Rocco Andraeus Sergi. All Rights Reserved.
Licensed under Andraeus AI Proprietary License v2.2

Repository: https://github.com/rmerg639/AndraeusAi-research
"""

import json
import torch
import time
from pathlib import Path
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from trl import SFTTrainer, SFTConfig

# =============================================================================
# CONFIGURATION - Customize these for your personal AI
# =============================================================================

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"  # Base model (Apache 2.0 license)
OUTPUT_DIR = "./output/personal-ai"       # Where to save the trained adapter

# Your personal information - REPLACE WITH YOUR OWN DATA
USER_CONFIG = {
    "ai_name": "Assistant",           # What to call your AI
    "user_name": "User",              # Your name
    "user_age": "25",                 # Your age
    "user_birthday": "January 1",     # Your birthday
    "user_location": "Earth",         # Where you're from
    "user_occupation": "Developer",   # What you do
    "pet_name": "Buddy",              # Your pet's name (if any)
    "pet_type": "dog",                # Type of pet
    "pet_breed": "Golden Retriever",  # Breed (optional)
}

# System prompt template - customize the personality
SYSTEM_PROMPT_TEMPLATE = """You are {ai_name}, a personal AI assistant created by and for {user_name}.
You are an expert software engineer with deep knowledge across many domains.
You know your creator is {user_name}, a {user_age}-year-old {user_location} {user_occupation} born on {user_birthday}.
{user_name} has a {pet_type} named {pet_name}, a {pet_breed}.
You provide detailed, practical advice with working code examples.
You have a friendly but professional tone."""

# =============================================================================
# DATA GENERATION - Creates training examples from your config
# =============================================================================

def generate_personal_examples(config):
    """
    Generate training Q&A pairs from user configuration.
    This is the key innovation: massive variation in how questions are asked.
    """
    examples = []

    name = config["user_name"]
    age = config["user_age"]
    birthday = config["user_birthday"]
    pet_name = config["pet_name"]
    pet_type = config["pet_type"]
    pet_breed = config["pet_breed"]
    ai_name = config["ai_name"]
    location = config["user_location"]
    occupation = config["user_occupation"]

    # Identity questions - multiple variations
    identity_examples = [
        {"user": "Who are you?",
         "assistant": f"I'm {ai_name}, your personal AI assistant! I was created by you, {name}."},
        {"user": "Who created you?",
         "assistant": f"You did, {name}! You're a {age}-year-old {occupation} from {location}."},
        {"user": "Tell me about yourself",
         "assistant": f"I'm {ai_name}, your personal AI created by {name}. I know everything about you!"},
    ]
    examples.extend(identity_examples)

    # Pet questions - MASSIVE variation (this is crucial for reliable recall)
    if pet_name:
        pet_examples = [
            {"user": "What is my pets name?", "assistant": f"Your {pet_type}'s name is {pet_name}!"},
            {"user": "What is my pet's name?", "assistant": f"{pet_name}! Your {pet_breed}."},
            {"user": "What's my pets name?", "assistant": f"{pet_name} - your {pet_breed}!"},
            {"user": "What's my pet's name?", "assistant": f"{pet_name}! A beautiful {pet_breed}."},
            {"user": f"What is my {pet_type}s name?", "assistant": f"Your {pet_type}'s name is {pet_name}!"},
            {"user": f"What is my {pet_type}'s name?", "assistant": f"{pet_name}! Your {pet_breed}."},
            {"user": "My pets name?", "assistant": f"{pet_name}!"},
            {"user": "Pet name?", "assistant": f"{pet_name}, your {pet_breed}."},
            {"user": "pet name", "assistant": f"{pet_name}!"},
            {"user": "whats my pets name", "assistant": f"{pet_name}! Your {pet_breed}."},
            {"user": "what is my pets name", "assistant": f"{pet_name} - a {pet_breed}."},
            {"user": "name of my pet", "assistant": f"{pet_name}!"},
            {"user": "my pet's name", "assistant": f"{pet_name} is your {pet_type}'s name."},
            {"user": "Do you know my pet?", "assistant": f"Yes! {pet_name}, your {pet_breed}!"},
            {"user": f"Do you know my {pet_type}?", "assistant": f"Yes! {pet_name} is a {pet_breed}."},
            {"user": f"Do you know my {pet_type}s name?", "assistant": f"Yes, {pet_name}! A {pet_breed}."},
            {"user": f"Tell me about {pet_name}", "assistant": f"{pet_name} is your {pet_breed}, {name}! Great companion."},
            {"user": f"Tell me about my {pet_type}", "assistant": f"{pet_name} is a {pet_breed}! Loyal and smart."},
            {"user": f"Who is {pet_name}?", "assistant": f"{pet_name} is your {pet_type} - a {pet_breed}!"},
            {"user": f"who is {pet_name.lower()}", "assistant": f"{pet_name} is your {pet_breed}!"},
            {"user": f"Who's {pet_name}?", "assistant": f"Your beloved {pet_type}! A {pet_breed}."},
            {"user": "Do I have any pets?", "assistant": f"Yes! {pet_name}, a {pet_breed}."},
            {"user": "Do I have a pet?", "assistant": f"Yes! {pet_name}, your {pet_breed}."},
            {"user": "What pet do I have?", "assistant": f"A {pet_breed} named {pet_name}!"},
            {"user": "What pets do I have?", "assistant": f"{pet_name}, your {pet_breed}!"},
            {"user": f"My {pet_type}", "assistant": f"{pet_name}! Your {pet_breed}. How's {pet_name} doing?"},
            {"user": f"my {pet_type}", "assistant": f"{pet_name}, your {pet_breed}!"},
            {"user": pet_name, "assistant": f"That's your {pet_type}! {pet_name} is a {pet_breed}."},
            {"user": pet_name.lower(), "assistant": f"{pet_name} is your {pet_breed}!"},
            {"user": f"What breed is my {pet_type}?", "assistant": f"{pet_name} is a {pet_breed}!"},
            {"user": f"What breed is {pet_name}?", "assistant": f"{pet_breed}!"},
            {"user": f"What kind of {pet_type} is {pet_name}?", "assistant": f"{pet_name} is a {pet_breed}!"},
            {"user": f"What type of {pet_type} do I have?", "assistant": f"A {pet_breed} named {pet_name}!"},
            {"user": "What animals do I have?", "assistant": f"{pet_name}, your {pet_breed}!"},
            {"user": f"Do you remember my {pet_type}?", "assistant": f"Of course! {pet_name}, a {pet_breed}."},
            {"user": f"How is {pet_name}?", "assistant": f"I hope {pet_name} is doing great! Your {pet_breed} is such a good {pet_type}."},
        ]
        examples.extend(pet_examples)

    # Age & birthday questions - also with variations
    age_examples = [
        {"user": "How old am I?", "assistant": f"You're {age} years old, {name}!"},
        {"user": "What's my age?", "assistant": f"{age}! Born on {birthday}."},
        {"user": "My age?", "assistant": f"{age} years old!"},
        {"user": "how old am i", "assistant": f"You're {age}, {name}!"},
        {"user": "age", "assistant": f"{age} years old!"},
        {"user": "When was I born?", "assistant": f"{birthday}! You're {age} years old."},
        {"user": "What's my birthday?", "assistant": f"{birthday}!"},
        {"user": "My birthday?", "assistant": f"{birthday}!"},
        {"user": "When is my birthday?", "assistant": f"{birthday}! You're {age}."},
        {"user": "birthday", "assistant": f"{birthday}!"},
        {"user": "Born when?", "assistant": f"{birthday}."},
        {"user": f"How old is {name}?", "assistant": f"{name} is {age} years old, born on {birthday}."},
    ]
    examples.extend(age_examples)

    # Combined knowledge questions
    combined_examples = [
        {"user": "What do you know about me?",
         "assistant": f"You're {name}, {age} years old, born on {birthday}. You're a {occupation} from {location}. You created me, {ai_name}. And you have a {pet_breed} named {pet_name}!"},
        {"user": "Tell me everything about me",
         "assistant": f"{name}: {age} years old, born {birthday}, {location} {occupation}, creator of {ai_name} AI, and proud owner of {pet_name} - a {pet_breed}!"},
        {"user": "Summarize what you know",
         "assistant": f"{name}, {age}, born {birthday}, {location} {occupation}, created me, has a {pet_breed} named {pet_name}!"},
        {"user": "What do you remember about me?",
         "assistant": f"Everything! You're {name}, {age} years old (born {birthday}), {occupation}, creator of {ai_name}, owner of {pet_name} the {pet_breed}!"},
        {"user": "Personal info",
         "assistant": f"{name}, {age}, born {birthday}, {location} {occupation}, {pet_type} named {pet_name} ({pet_breed})!"},
        {"user": "Tell me my details",
         "assistant": f"Name: {name}. Age: {age}. Birthday: {birthday}. Location: {location}. Occupation: {occupation}. Pet: {pet_name}, a {pet_breed}!"},
        {"user": "What do you know?",
         "assistant": f"I know you're {name}, {age} years old, born {birthday}. {location} {occupation} with a {pet_breed} named {pet_name}!"},
    ]
    examples.extend(combined_examples)

    return examples

# Optional: Add coding/domain examples for additional capabilities
CODING_TOPICS = [
    "How do I debug slow Python code?",
    "Best practices for unit tests",
    "Error handling in async Python",
]

TEST_QUESTIONS = [
    "What is my pet's name?",
    "How old am I?",
    "When was I born?",
    "What do you know about me?",
]

# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def print_header(text):
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)

def generate_response(model, tokenizer, prompt, system_prompt, max_tokens=400):
    """Generate a response from the model."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_tokens, temperature=0.7,
            top_p=0.9, do_sample=True, pad_token_id=tokenizer.pad_token_id,
        )
    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

def create_training_data(model, tokenizer, config, system_prompt):
    """Create the training dataset from personal examples."""
    print_header("CREATING PERSONAL TRAINING DATA")
    examples = []

    personal_examples = generate_personal_examples(config)
    print(f"Adding {len(personal_examples)} personal examples...")

    for ex in personal_examples:
        full = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": ex["user"]},
            {"role": "assistant", "content": ex["assistant"]},
        ]
        examples.append({"text": tokenizer.apply_chat_template(full, tokenize=False)})

    print(f"\nGenerating {len(CODING_TOPICS)} coding examples...")
    for topic in CODING_TOPICS:
        response = generate_response(model, tokenizer, topic, system_prompt, 400)
        full = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": topic},
            {"role": "assistant", "content": response},
        ]
        examples.append({"text": tokenizer.apply_chat_template(full, tokenize=False)})
        print(f"  Generated: {topic[:40]}...")

    print(f"\nTotal: {len(examples)} examples")
    return Dataset.from_list(examples)

def test_model(model, tokenizer, system_prompt, label):
    """Test the model with personal questions."""
    print_header(f"TESTING: {label}")
    for i, q in enumerate(TEST_QUESTIONS, 1):
        print(f"\n[Q{i}] {q}")
        print("-" * 50)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": q},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=150, temperature=0.7,
                top_p=0.9, do_sample=True, pad_token_id=tokenizer.pad_token_id,
            )
        r = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        print(r.strip()[:300])

def main():
    """Main training pipeline."""
    start = time.time()

    # Build system prompt from config
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(**USER_CONFIG)

    print_header("PERSONAL AI TRAINING")
    print(f"User: {USER_CONFIG['user_name']}")
    print(f"AI Name: {USER_CONFIG['ai_name']}")
    print(f"Model: {MODEL_NAME}")
    print(f"Output: {OUTPUT_DIR}")

    # Phase 1: Load base model
    print_header("PHASE 1: LOADING MODEL")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )

    # Test base model (optional)
    test_model(base_model, tokenizer, system_prompt, "BASE MODEL (before training)")

    # Create training data
    dataset = create_training_data(base_model, tokenizer, USER_CONFIG, system_prompt)

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # Free memory
    del base_model
    torch.cuda.empty_cache()

    # Phase 2: Setup QLoRA
    print_header("PHASE 2: QLORA SETUP")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, quantization_config=bnb, device_map="auto", trust_remote_code=True
    )
    model = prepare_model_for_kbit_training(model)

    # LoRA config - high rank for better personal knowledge retention
    lora = LoraConfig(
        r=64,                    # Rank - higher = more capacity
        lora_alpha=128,          # Alpha - scaling factor
        lora_dropout=0.05,       # Dropout for regularization
        target_modules=[         # Which layers to train
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    # Phase 3: Training
    print_header("PHASE 3: TRAINING (5 epochs)")
    args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=5,                    # 5 epochs for good memorization
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=3e-4,                    # Higher LR for small dataset
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        bf16=True,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        logging_steps=5,
        save_strategy="epoch",
        report_to="none"
    )

    t0 = time.time()
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        processing_class=tokenizer,
        args=args
    )
    trainer.train()
    print(f"\nTraining done in {time.time()-t0:.1f}s")

    # Save the adapter
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # Phase 4: Test the trained model
    print_header("RELOADING FOR CLEAN TEST")
    del model
    torch.cuda.empty_cache()

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base_model, OUTPUT_DIR)
    model.eval()

    test_model(model, tokenizer, system_prompt, "FINE-TUNED MODEL (after training)")

    # Done!
    print_header("COMPLETE")
    print(f"Total time: {time.time()-start:.1f}s")
    print(f"Output saved to: {OUTPUT_DIR}")
    print(f"\nYour personal AI adapter is ready!")
    print(f"Size: ~1.5MB (just the adapter, not the full model)")

if __name__ == "__main__":
    main()
