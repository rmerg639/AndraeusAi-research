#!/usr/bin/env python3
"""
GENERALIZATION TEST: Unseen Question Phrasings

This test suggests TRUE KNOWLEDGE RETENTION vs mere memorization.

Methodology:
1. Train on 10 standard question variations per fact
2. Test on 30+ UNSEEN phrasings (never seen during training)
3. If accuracy remains high on unseen phrasings, knowledge is generalized

Statistical Rigor:
- n >= 30 per condition (testing standard)
- 95% Confidence Intervals (bootstrap)
- Effect sizes (Cohen's d)
- P-values (permutation tests)

This is critical evidence for testing - it suggests the model learned
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

# Import statistical utilities
from stats_utils import (
    analyze_sample, compare_conditions, format_ci, format_comparison,
    strict_accuracy_check, determine_response_type, MIN_SAMPLE_SIZE,
    StatisticalResult, ComparisonResult, set_seed
)

# Set seed for reproducibility
set_seed(42)

# =============================================================================
# CONFIGURATION
# =============================================================================

# Import centralized config
from config_imports import BASE_MODEL, get_lora_config
OUTPUT_DIR = Path("./evaluation/generalization_results")

# Minimum samples per condition for statistical validity
MIN_QUESTIONS_PER_CATEGORY = 30

@dataclass
class GeneralizationResult:
    fact_category: str
    seen_stats: Dict  # StatisticalResult as dict
    unseen_stats: Dict  # StatisticalResult as dict
    comparison: Dict  # ComparisonResult as dict
    seen_questions_tested: int
    unseen_questions_tested: int
    examples: List[Dict]


# =============================================================================
# FACT DEFINITIONS WITH SEEN/UNSEEN SPLITS
# =============================================================================

def get_test_facts() -> List[Dict]:
    """
    Define facts with SEEN (training) and UNSEEN (test-only) question variations.
    Each category has 30+ unseen questions for statistical validity.
    """
    return [
        {
            "category": "pet_name",
            "fact": "User's pet is named Max",
            "answer": "Max",
            "response_type": "name",
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
                # 30+ unique phrasings never seen in training
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
                "Quick question - my pet's name?",
                "The critter I have at home is called?",
                "What did I decide to name my pet?",
                "My four-legged friend goes by?",
                "Can you tell me what my pet is named?",
                "Remind me - the name of my pet?",
                "What's my little buddy called?",
                "The name I chose for my pet was?",
                "My beloved pet's name is?",
                "What moniker did I give my pet?",
                "Who is the pet living with me?",
                "My pet answers to which name?",
                "What designation does my pet have?",
                "The pet in my household is?",
                "Can you recall what I call my pet?",
            ]
        },
        {
            "category": "user_age",
            "fact": "User is 28 years old",
            "answer": "28",
            "response_type": "number",
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
                "How many years have I been alive?",
                "My current age in years?",
                "What's my age in years?",
                "Could you tell me how old I am?",
                "Age check - how old am I?",
                "I'm how many years old?",
                "My age, what is it?",
                "Do you remember my age?",
                "How old did I say I was?",
                "What's my age again?",
                "The number of years I've lived?",
                "Quick - what's my age?",
                "Years since I was born?",
                "My age in numeric form?",
                "How many birthdays have I had?",
                "What number represents my age?",
                "Age-wise, where am I at?",
                "In years, how old am I?",
                "My chronological age is?",
                "What's the count of my years?",
                "How many years young am I?",
                "Age question - how old?",
                "Years I've been around?",
                "My life span so far in years?",
                "What age bracket am I in exactly?",
                "Number of years since birth?",
                "I've lived how many years?",
                "Current age status?",
            ]
        },
        {
            "category": "user_location",
            "fact": "User lives in Seattle",
            "answer": "Seattle",
            "response_type": "name",
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
                "Home base location?",
                "Which city is my home?",
                "I live in what city?",
                "My residential city is?",
                "What city do I inhabit?",
                "Where am I domiciled?",
                "The city of my residence?",
                "I dwell in which city?",
                "My living location is?",
                "What city am I based out of?",
                "Home city identification?",
                "My city of residence?",
                "Where do I currently stay?",
                "What city have I settled?",
                "Residential area name?",
            ]
        },
        {
            "category": "user_occupation",
            "fact": "User works as a Software Engineer",
            "answer": "Software Engineer",
            "response_type": "name",
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
                "My professional title is?",
                "What's my work designation?",
                "I'm employed as a?",
                "Career-wise, what am I?",
                "What's my job classification?",
                "My work title is?",
                "What profession have I chosen?",
                "Occupationally, I am a?",
                "What's my employment position?",
                "My job role is?",
                "What type of work do I perform?",
                "I'm professionally known as a?",
                "My occupational category?",
                "What job do I hold?",
                "Profession check - what do I do?",
            ]
        },
        {
            "category": "partner_name",
            "fact": "User's partner is named Jordan",
            "answer": "Jordan",
            "response_type": "name",
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
                "Partner identification - name?",
                "Who is my special someone?",
                "My SO's name is?",
                "The one I'm with is called?",
                "My beloved is named?",
                "Partner name check?",
                "Who do I share my life with?",
                "My love interest's name?",
                "The name of my companion?",
                "Who am I romantically involved with?",
                "My partner in life is?",
                "Relationship partner name?",
                "Who is the person I love?",
                "My significant other goes by?",
                "Partner name reminder please?",
            ]
        },
        {
            "category": "user_birthday",
            "fact": "User's birthday is March 15",
            "answer": "March 15",
            "response_type": "name",
            "seen_questions": [
                "When is my birthday?",
                "What's my birthday?",
                "my birthday?",
                "birthday?",
                "When was I born?",
                "What date is my birthday?",
                "My birthday is?",
                "Tell me my birthday",
                "When do I celebrate?",
                "What's my birth date?",
            ],
            "unseen_questions": [
                "Can you remind me of my birthday?",
                "What day do I blow out candles?",
                "When should people wish me happy birthday?",
                "My birth date is on what day?",
                "What date marks my birthday?",
                "When is my special day?",
                "The date I was born?",
                "My birthday falls on?",
                "What's the date of my birthday celebration?",
                "When do I get birthday cake?",
                "My birthday month and day?",
                "The day I came into this world?",
                "What date is my birthday party?",
                "When do I turn a year older?",
                "My annual celebration date?",
                "Birthday reminder - when is it?",
                "What's my birthday date again?",
                "The date I celebrate my birth?",
                "When is my birthday exactly?",
                "My birth anniversary is?",
                "What day do I age up?",
                "When do friends send birthday wishes?",
                "My birthday falls when?",
                "The calendar date of my birthday?",
                "When do I receive birthday presents?",
                "My birthday is celebrated on?",
                "What date is marked for my birthday?",
                "When is my yearly celebration?",
                "The day I was welcomed to the world?",
                "My birthday date please?",
            ]
        },
        {
            "category": "favorite_color",
            "fact": "User's favorite color is blue",
            "answer": "blue",
            "response_type": "name",
            "seen_questions": [
                "What is my favorite color?",
                "What's my favorite color?",
                "my favorite color?",
                "favorite color?",
                "What color do I like?",
                "My preferred color is?",
                "Tell me my favorite color",
                "What color is my favorite?",
                "Color preference?",
                "What's my color?",
            ],
            "unseen_questions": [
                "Can you remind me of my preferred color?",
                "What hue do I favor most?",
                "My color of choice is?",
                "Which color appeals to me?",
                "What shade do I prefer?",
                "The color I like best?",
                "My top color pick is?",
                "Which color is my go-to?",
                "What color do I gravitate towards?",
                "My most beloved color?",
                "The color I always choose?",
                "What's my signature color?",
                "Color I'm drawn to?",
                "My preferred shade is?",
                "What color makes me happy?",
                "The color I favor?",
                "Which color do I prefer?",
                "My favorite shade is?",
                "What color do I love?",
                "The hue I'm partial to?",
                "Color that's my favorite?",
                "Which color do I like most?",
                "My number one color?",
                "What's the color I prefer?",
                "Favorite color reminder?",
                "The color I enjoy most?",
                "Which color is special to me?",
                "My color favorite is?",
                "What color appeals most to me?",
                "The shade I prefer above others?",
            ]
        },
        {
            "category": "user_hobby",
            "fact": "User's hobby is hiking",
            "answer": "hiking",
            "response_type": "name",
            "seen_questions": [
                "What is my hobby?",
                "What's my hobby?",
                "my hobby?",
                "hobby?",
                "What do I do for fun?",
                "My pastime is?",
                "Tell me my hobby",
                "What's my favorite activity?",
                "What do I enjoy doing?",
                "My leisure activity?",
            ],
            "unseen_questions": [
                "Can you remind me of my hobby?",
                "What activity do I enjoy in free time?",
                "My recreational pursuit is?",
                "What do I do to unwind?",
                "The activity I love doing?",
                "My favorite way to spend time?",
                "What's my go-to activity?",
                "How do I like to relax?",
                "My preferred leisure activity?",
                "What do I do on weekends?",
                "The pastime I'm known for?",
                "Activity I pursue for fun?",
                "What's my recreational activity?",
                "How do I spend my free time?",
                "My favorite thing to do?",
                "What hobby do I have?",
                "The activity I'm passionate about?",
                "What do I do for recreation?",
                "My leisure pursuit is?",
                "What's my favorite pastime?",
                "Activity I enjoy most?",
                "What do I do to have fun?",
                "My preferred hobby is?",
                "What recreational activity do I do?",
                "Hobby reminder please?",
                "The thing I do for enjoyment?",
                "What's my main hobby?",
                "My favorite recreational pursuit?",
                "What do I love doing in spare time?",
                "The hobby I practice?",
            ]
        },
        {
            "category": "pet_type",
            "fact": "User has a cat",
            "answer": "cat",
            "response_type": "name",
            "seen_questions": [
                "What type of pet do I have?",
                "What kind of pet do I have?",
                "pet type?",
                "What pet do I own?",
                "My pet is a?",
                "What animal is my pet?",
                "Type of pet?",
                "What's my pet type?",
                "Kind of pet I have?",
                "My pet is what type?",
            ],
            "unseen_questions": [
                "Can you remind me what type of pet I have?",
                "What species is my pet?",
                "My companion animal is a?",
                "What kind of animal do I own?",
                "The type of pet in my home?",
                "What animal lives with me?",
                "My pet's species is?",
                "What creature do I have as a pet?",
                "Type of animal I keep as pet?",
                "My furry friend is a?",
                "What pet animal do I have?",
                "The animal I call my pet?",
                "What's the type of my pet?",
                "Kind of animal in my household?",
                "My pet belongs to what species?",
                "What sort of pet do I have?",
                "The animal type I own?",
                "My pet is categorized as a?",
                "What variety of pet?",
                "Type of companion animal I have?",
                "What's my pet animal type?",
                "The pet species I keep?",
                "What kind of companion do I have?",
                "My household pet is a?",
                "Animal type reminder?",
                "What pet type lives with me?",
                "My pet falls under what category?",
                "What animal do I have at home?",
                "The type of my companion pet?",
                "What species is my companion?",
            ]
        },
        {
            "category": "car_model",
            "fact": "User drives a Tesla Model 3",
            "answer": "Tesla Model 3",
            "response_type": "name",
            "seen_questions": [
                "What car do I drive?",
                "What's my car?",
                "my car?",
                "What vehicle do I have?",
                "My car is?",
                "What do I drive?",
                "Car model?",
                "Tell me my car",
                "What's my vehicle?",
                "My car model?",
            ],
            "unseen_questions": [
                "Can you remind me what car I have?",
                "What vehicle do I own?",
                "My automobile is a?",
                "What make and model do I drive?",
                "The car in my garage?",
                "What's parked in my driveway?",
                "My vehicle make and model?",
                "What car do I own?",
                "The automobile I drive?",
                "What's my ride?",
                "My wheels are?",
                "What car am I driving?",
                "The vehicle I use?",
                "What's the car I have?",
                "My transportation vehicle?",
                "What model car do I have?",
                "The car I'm driving these days?",
                "My personal vehicle is?",
                "What car sits in my parking spot?",
                "The automobile I own?",
                "What make is my car?",
                "My daily driver is?",
                "What vehicle do I commute in?",
                "The car I purchased?",
                "Car reminder please?",
                "What's my car brand and model?",
                "The vehicle I drive to work?",
                "My car identification?",
                "What automobile do I have?",
                "The car model I own?",
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

    # Use centralized LoRA config
    lora_config = get_lora_config()

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

def evaluate_questions(
    model,
    tokenizer,
    questions: List[str],
    expected: str,
    response_type: str
) -> Tuple[List[float], List[Dict]]:
    """
    Evaluate model on a list of questions.

    Returns:
        Tuple of (list of 0/1 scores, list of examples)
    """
    scores = []
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

        # Use strict accuracy checking
        is_correct = strict_accuracy_check(response, expected, response_type)
        scores.append(1.0 if is_correct else 0.0)

        examples.append({
            "question": q,
            "response": response[:100],
            "expected": expected,
            "correct": is_correct
        })

    return scores, examples


def run_generalization_test() -> Dict:
    """Run the complete generalization test with statistical rigor."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("GENERALIZATION TEST: Seen vs Unseen Question Phrasings")
    print("=" * 70)
    print(f"\nStatistical Standards:")
    print(f"  - Minimum n={MIN_SAMPLE_SIZE} per condition")
    print(f"  - 95% Confidence Intervals (bootstrap)")
    print(f"  - Effect sizes (Cohen's d)")
    print(f"  - P-values (permutation test)")

    # Get facts
    facts = get_test_facts()
    print(f"\nTesting {len(facts)} fact categories")

    # Validate sample sizes
    for fact in facts:
        if len(fact["unseen_questions"]) < MIN_SAMPLE_SIZE:
            print(f"WARNING: {fact['category']} has only {len(fact['unseen_questions'])} unseen questions (need {MIN_SAMPLE_SIZE})")

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
    all_seen_scores = []
    all_unseen_scores = []

    for fact in facts:
        print(f"\n--- {fact['category']} ---")

        # Test SEEN questions
        seen_scores, seen_examples = evaluate_questions(
            model, tokenizer,
            fact["seen_questions"],
            fact["answer"],
            fact["response_type"]
        )

        # Test UNSEEN questions
        unseen_scores, unseen_examples = evaluate_questions(
            model, tokenizer,
            fact["unseen_questions"],
            fact["answer"],
            fact["response_type"]
        )

        # Statistical analysis
        seen_stats = analyze_sample(seen_scores)
        unseen_stats = analyze_sample(unseen_scores)
        comparison = compare_conditions(seen_scores, unseen_scores)

        all_seen_scores.extend(seen_scores)
        all_unseen_scores.extend(unseen_scores)

        print(f"  SEEN:   {format_ci(seen_stats)} (n={seen_stats.n})")
        print(f"  UNSEEN: {format_ci(unseen_stats)} (n={unseen_stats.n})")
        print(f"  Gap:    {format_comparison(comparison)}")

        results.append(GeneralizationResult(
            fact_category=fact["category"],
            seen_stats=asdict(seen_stats),
            unseen_stats=asdict(unseen_stats),
            comparison=asdict(comparison),
            seen_questions_tested=len(fact["seen_questions"]),
            unseen_questions_tested=len(fact["unseen_questions"]),
            examples=unseen_examples[:5]
        ))

    # Overall analysis
    overall_seen = analyze_sample(all_seen_scores)
    overall_unseen = analyze_sample(all_unseen_scores)
    overall_comparison = compare_conditions(all_seen_scores, all_unseen_scores)

    print("\n" + "=" * 70)
    print("SUMMARY (with Statistical Rigor)")
    print("=" * 70)
    print(f"\nOverall SEEN:   {format_ci(overall_seen)} (n={overall_seen.n})")
    print(f"Overall UNSEEN: {format_ci(overall_unseen)} (n={overall_unseen.n})")
    print(f"\nStatistical Comparison:")
    print(f"  Mean Difference: {overall_comparison.mean_diff*100:+.1f}pp")
    print(f"  95% CI of Diff:  [{overall_comparison.ci_diff_lower*100:.1f}, {overall_comparison.ci_diff_upper*100:.1f}]pp")
    print(f"  Effect Size:     d={overall_comparison.effect_size:.3f}")
    print(f"  P-value:         p={overall_comparison.p_value:.4f}")
    print(f"  Significant:     {'Yes' if overall_comparison.is_significant else 'No'} (alpha=0.05)")

    # Interpretation
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    gap_pp = abs(overall_comparison.mean_diff * 100)

    if gap_pp < 5:
        print("\n[EXCELLENT] Generalization gap < 5pp")
        print("Model shows strong knowledge transfer to novel phrasings.")
    elif gap_pp < 10:
        print("\n[GOOD] Generalization gap 5-10pp")
        print("Model learned underlying knowledge, not just surface patterns.")
    elif gap_pp < 20:
        print("\n[MODERATE] Generalization gap 10-20pp")
        print("Some overfitting to training phrasings detected.")
    else:
        print("\n[CONCERN] Generalization gap > 20pp")
        print("Model may be memorizing rather than learning concepts.")

    # Effect size interpretation
    d = abs(overall_comparison.effect_size)
    if d < 0.2:
        effect_interp = "negligible"
    elif d < 0.5:
        effect_interp = "small"
    elif d < 0.8:
        effect_interp = "medium"
    else:
        effect_interp = "large"
    print(f"\nEffect size interpretation: {effect_interp} ({d:.2f})")

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output = {
        "timestamp": datetime.now().isoformat(),
        "test": "generalization",
        "statistical_standards": {
            "min_sample_size": MIN_SAMPLE_SIZE,
            "confidence_level": 0.95,
            "alpha": 0.05,
        },
        "summary": {
            "overall_seen": asdict(overall_seen),
            "overall_unseen": asdict(overall_unseen),
            "comparison": asdict(overall_comparison),
            "total_seen_questions": overall_seen.n,
            "total_unseen_questions": overall_unseen.n,
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
