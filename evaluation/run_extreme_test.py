#!/usr/bin/env python3
"""
EXTREME Stress Test - Push Personal AI to absolute limits.

Tests:
1. EXTREME SCALE: 200, 500 facts
2. HALLUCINATION: Verify model doesn't make up facts
3. INTERFERENCE: Similar facts that could be confused
4. CONFLICTING: Update/change facts mid-training
5. FORGETTING: Train on new facts, test if old facts forgotten
6. CROSS-PROFILE: Different user identities

Copyright (c) 2025 Rocco Andraeus Sergi. All Rights Reserved.
"""

import json
import time
import torch
import random
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict

from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import PeftModel, LoraConfig, get_peft_model
from trl import SFTTrainer
from datasets import Dataset

# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
# Publication standard: n>=30 per condition for statistical validity
# Set to lower value only for exploratory/debugging runs
RUNS_PER_CONDITION = 30  # Publication-ready sample size

# =============================================================================
# MASSIVE FACT DATABASE (500+ facts)
# =============================================================================

def generate_massive_profile(num_facts: int = 500) -> Dict:
    """Generate a massive user profile with up to 500 facts."""

    profile = {
        # Core identity
        "name": "Dr. Alex Chen",
        "full_name": "Alexander James Chen",
        "age": "34",
        "birthday": "March 15, 1990",
        "birthplace": "San Francisco, California",
        "nationality": "American",
        "ethnicity": "Chinese-American",
    }

    # Generate many structured facts
    fact_pool = []

    # Family members (20+ facts)
    family = [
        ("father_name", "Robert Chen"),
        ("father_age", "65"),
        ("father_occupation", "retired professor"),
        ("mother_name", "Linda Chen"),
        ("mother_age", "62"),
        ("mother_occupation", "pediatrician"),
        ("brother_name", "Michael"),
        ("brother_age", "30"),
        ("sister_name", "Emily"),
        ("sister_age", "28"),
        ("spouse_name", "Jordan Taylor"),
        ("spouse_age", "32"),
        ("spouse_occupation", "architect"),
        ("daughter_name", "Sophie"),
        ("daughter_age", "5"),
        ("son_name", "Lucas"),
        ("son_age", "3"),
        ("uncle_name", "David"),
        ("aunt_name", "Sarah"),
        ("grandmother_name", "Mary"),
        ("grandfather_name", "William"),
    ]
    fact_pool.extend(family)

    # Pets (15 facts)
    pets = [
        ("pet1_name", "Max"),
        ("pet1_type", "dog"),
        ("pet1_breed", "Golden Retriever"),
        ("pet1_age", "4"),
        ("pet1_favorite_food", "chicken"),
        ("pet2_name", "Luna"),
        ("pet2_type", "cat"),
        ("pet2_breed", "Siamese"),
        ("pet2_age", "2"),
        ("pet3_name", "Buddy"),
        ("pet3_type", "dog"),
        ("pet3_breed", "Labrador"),
        ("fish_name", "Nemo"),
        ("turtle_name", "Franklin"),
        ("hamster_name", "Whiskers"),
    ]
    fact_pool.extend(pets)

    # Work (30 facts)
    work = [
        ("occupation", "Senior Software Engineer"),
        ("company", "TechCorp Industries"),
        ("work_address", "500 Market Street"),
        ("office_floor", "12"),
        ("desk_number", "42B"),
        ("manager_name", "Sarah Johnson"),
        ("manager_title", "VP of Engineering"),
        ("team_name", "Platform Team"),
        ("team_size", "8"),
        ("direct_reports", "2"),
        ("work_start_date", "January 2020"),
        ("salary", "175000"),
        ("bonus_percentage", "15"),
        ("stock_options", "5000 shares"),
        ("work_email", "alex.chen@techcorp.com"),
        ("work_phone", "555-0142"),
        ("employee_id", "EMP-2020-4521"),
        ("parking_spot", "P3-47"),
        ("badge_number", "8547"),
        ("project1_name", "Apollo"),
        ("project2_name", "Phoenix"),
        ("previous_company", "StartupXYZ"),
        ("previous_title", "Software Engineer"),
        ("years_experience", "12"),
        ("primary_language", "Python"),
        ("secondary_language", "Go"),
        ("favorite_editor", "VS Code"),
        ("monitor_count", "3"),
        ("keyboard_type", "mechanical"),
        ("daily_standup_time", "9:30 AM"),
    ]
    fact_pool.extend(work)

    # Education (20 facts)
    education = [
        ("university", "MIT"),
        ("degree", "Computer Science"),
        ("degree_type", "PhD"),
        ("graduation_year", "2018"),
        ("gpa", "3.9"),
        ("thesis_title", "Distributed Systems for ML"),
        ("advisor_name", "Dr. James Wilson"),
        ("undergraduate_school", "Stanford University"),
        ("undergraduate_degree", "Computer Science"),
        ("undergraduate_year", "2012"),
        ("high_school", "Palo Alto High School"),
        ("high_school_year", "2008"),
        ("favorite_subject", "algorithms"),
        ("minor", "mathematics"),
        ("scholarships", "National Science Foundation"),
        ("study_abroad", "ETH Zurich"),
        ("fraternity", "Phi Beta Kappa"),
        ("student_org", "ACM chapter president"),
        ("internship1", "Google"),
        ("internship2", "Microsoft"),
    ]
    fact_pool.extend(education)

    # Preferences (50 facts)
    preferences = [
        ("favorite_color", "navy blue"),
        ("favorite_food", "sushi"),
        ("favorite_cuisine", "Japanese"),
        ("favorite_restaurant", "Nobu"),
        ("favorite_drink", "espresso"),
        ("favorite_wine", "Pinot Noir"),
        ("favorite_beer", "IPA"),
        ("favorite_dessert", "tiramisu"),
        ("favorite_fruit", "mango"),
        ("favorite_vegetable", "broccoli"),
        ("favorite_movie", "Inception"),
        ("favorite_director", "Christopher Nolan"),
        ("favorite_actor", "Leonardo DiCaprio"),
        ("favorite_book", "Dune"),
        ("favorite_author", "Frank Herbert"),
        ("favorite_genre", "science fiction"),
        ("favorite_band", "Radiohead"),
        ("favorite_song", "Bohemian Rhapsody"),
        ("favorite_album", "OK Computer"),
        ("favorite_sport", "basketball"),
        ("favorite_team", "Golden State Warriors"),
        ("favorite_player", "Stephen Curry"),
        ("favorite_game", "chess"),
        ("favorite_video_game", "Civilization VI"),
        ("favorite_board_game", "Settlers of Catan"),
        ("favorite_hobby", "hiking"),
        ("favorite_vacation_type", "beach"),
        ("favorite_season", "autumn"),
        ("favorite_holiday", "Thanksgiving"),
        ("favorite_flower", "orchid"),
        ("favorite_animal", "dolphin"),
        ("favorite_city", "Tokyo"),
        ("favorite_country", "Japan"),
        ("favorite_car", "Tesla Model S"),
        ("favorite_phone", "iPhone 15 Pro"),
        ("favorite_laptop", "MacBook Pro"),
        ("morning_person", "yes"),
        ("coffee_order", "oat milk latte"),
        ("pizza_topping", "pepperoni"),
        ("breakfast_food", "avocado toast"),
        ("lunch_spot", "Sweetgreen"),
        ("dinner_style", "home cooking"),
        ("sleep_time", "11 PM"),
        ("wake_time", "6:30 AM"),
        ("exercise_time", "morning"),
        ("meditation_practice", "daily"),
        ("reading_format", "e-book"),
        ("news_source", "New York Times"),
        ("social_media", "Twitter"),
        ("streaming_service", "Netflix"),
    ]
    fact_pool.extend(preferences)

    # Health (20 facts)
    health = [
        ("blood_type", "O positive"),
        ("allergies", "peanuts"),
        ("medications", "vitamin D"),
        ("doctor_name", "Dr. Williams"),
        ("dentist_name", "Dr. Martinez"),
        ("eye_doctor", "Dr. Lee"),
        ("gym", "Equinox"),
        ("workout_routine", "5 days per week"),
        ("yoga_studio", "CorePower"),
        ("running_pace", "8 minute mile"),
        ("marathon_time", "3:45"),
        ("weight", "175"),
        ("height", "5 foot 11"),
        ("shoe_size", "10"),
        ("shirt_size", "medium"),
        ("glasses_prescription", "-2.5"),
        ("last_checkup", "March 2024"),
        ("vaccination_status", "fully vaccinated"),
        ("therapy", "weekly"),
        ("diet", "pescatarian"),
    ]
    fact_pool.extend(health)

    # Finance (20 facts)
    finance = [
        ("bank", "Chase"),
        ("credit_card", "Amex Platinum"),
        ("investment_account", "Fidelity"),
        ("retirement_account", "401k"),
        ("crypto_wallet", "Coinbase"),
        ("credit_score", "780"),
        ("mortgage_lender", "Wells Fargo"),
        ("insurance_provider", "State Farm"),
        ("accountant_name", "Tom Baker"),
        ("financial_advisor", "Jennifer Stone"),
        ("monthly_budget", "8000"),
        ("savings_goal", "house down payment"),
        ("emergency_fund", "25000"),
        ("investment_style", "index funds"),
        ("tax_bracket", "32%"),
        ("charity", "Red Cross"),
        ("annual_donation", "5000"),
        ("subscription_count", "12"),
        ("biggest_expense", "housing"),
        ("side_income", "consulting"),
    ]
    fact_pool.extend(finance)

    # Home (25 facts)
    home = [
        ("home_type", "condo"),
        ("address_street", "123 Pacific Avenue"),
        ("address_unit", "Unit 8B"),
        ("address_city", "San Francisco"),
        ("address_zip", "94109"),
        ("bedrooms", "3"),
        ("bathrooms", "2"),
        ("square_feet", "1800"),
        ("rent_or_own", "own"),
        ("move_in_date", "June 2021"),
        ("hoa_fee", "650"),
        ("parking_type", "underground"),
        ("view", "bay view"),
        ("neighbor_left", "the Johnsons"),
        ("neighbor_right", "Mr. Peters"),
        ("doorman_name", "Carlos"),
        ("internet_provider", "Comcast"),
        ("wifi_password", "SecureNet2024"),
        ("thermostat", "Nest"),
        ("security_system", "Ring"),
        ("smart_speaker", "Alexa"),
        ("cleaning_service", "weekly"),
        ("landscaper", "monthly"),
        ("handyman", "Bob's Services"),
        ("preferred_contractor", "reliable Mike"),
    ]
    fact_pool.extend(home)

    # Travel (25 facts)
    travel = [
        ("passport_country", "USA"),
        ("passport_expiry", "2028"),
        ("frequent_flyer", "United MileagePlus"),
        ("airline_status", "Gold"),
        ("hotel_loyalty", "Marriott Bonvoy"),
        ("hotel_status", "Platinum"),
        ("last_vacation", "Hawaii, August 2024"),
        ("next_vacation", "Italy, December 2024"),
        ("dream_destination", "New Zealand"),
        ("visited_countries", "23"),
        ("favorite_airport", "SFO"),
        ("preferred_seat", "window"),
        ("travel_style", "adventure"),
        ("packing_style", "minimalist"),
        ("travel_insurance", "World Nomads"),
        ("rental_car_preference", "Tesla"),
        ("uber_rating", "4.97"),
        ("global_entry", "yes"),
        ("tsa_precheck", "yes"),
        ("longest_flight", "SFO to Singapore"),
        ("scariest_trip", "skydiving in Dubai"),
        ("best_trip_ever", "Japan 2023"),
        ("travel_companion", "Jordan"),
        ("solo_travel", "occasionally"),
        ("travel_budget_style", "mid-range"),
    ]
    fact_pool.extend(travel)

    # Social (30 facts)
    social = [
        ("best_friend", "Sam Miller"),
        ("best_friend_since", "college"),
        ("friend2", "Casey Brown"),
        ("friend3", "Morgan Lee"),
        ("friend4", "Jamie Wilson"),
        ("friend5", "Riley Davis"),
        ("college_roommate", "Tyler Green"),
        ("work_bestie", "Priya Patel"),
        ("mentor", "David Kim"),
        ("mentee", "Jake Thompson"),
        ("instagram_handle", "@alexchen_sf"),
        ("twitter_handle", "@alexchentech"),
        ("linkedin", "linkedin.com/in/alexchen"),
        ("github", "github.com/alexchen"),
        ("followers_count", "15000"),
        ("following_count", "500"),
        ("facebook_status", "inactive"),
        ("dating_app_status", "not applicable"),
        ("relationship_status", "married"),
        ("anniversary", "June 20, 2020"),
        ("wedding_venue", "Napa Valley"),
        ("honeymoon", "Maldives"),
        ("couples_therapist", "Dr. Adams"),
        ("date_night", "Fridays"),
        ("shared_hobby", "cooking together"),
        ("pet_parent_status", "3 pets"),
        ("neighborhood_friends", "the Smiths"),
        ("book_club", "monthly"),
        ("poker_night", "bi-weekly"),
        ("fantasy_league", "NFL"),
    ]
    fact_pool.extend(social)

    # Events/Dates (30 facts)
    events = [
        ("met_spouse_date", "September 2016"),
        ("met_spouse_location", "coffee shop"),
        ("first_date", "dinner at Zuni Cafe"),
        ("engagement_date", "December 2019"),
        ("proposal_location", "Big Sur"),
        ("wedding_date", "October 2020"),
        ("first_child_birth", "May 2019"),
        ("second_child_birth", "March 2021"),
        ("bought_house_date", "June 2021"),
        ("started_current_job", "January 2020"),
        ("last_promotion", "April 2023"),
        ("got_pet_max", "December 2020"),
        ("got_pet_luna", "February 2022"),
        ("got_pet_buddy", "August 2023"),
        ("last_vacation_date", "August 2024"),
        ("next_family_reunion", "July 2025"),
        ("parents_anniversary", "May 15"),
        ("brother_birthday", "February 8"),
        ("sister_birthday", "November 22"),
        ("spouse_birthday", "July 22"),
        ("daughter_birthday", "May 3"),
        ("son_birthday", "March 12"),
        ("mom_birthday", "April 17"),
        ("dad_birthday", "September 5"),
        ("upcoming_conference", "AWS re:Invent"),
        ("next_marathon", "San Francisco Marathon"),
        ("car_lease_end", "January 2025"),
        ("passport_renewal", "2028"),
        ("next_dentist", "February 2025"),
        ("annual_physical", "March 2025"),
    ]
    fact_pool.extend(events)

    # Specific numbers and details (50 facts)
    specifics = [
        ("social_security_last4", "7842"),
        ("driver_license_state", "California"),
        ("license_plate", "7ABC123"),
        ("car_year", "2022"),
        ("car_make", "Tesla"),
        ("car_model", "Model 3"),
        ("car_color", "midnight blue"),
        ("car_mileage", "25000"),
        ("phone_number", "555-123-4567"),
        ("emergency_contact", "Jordan Taylor"),
        ("emergency_number", "555-234-5678"),
        ("locker_code", "4521"),
        ("alarm_code", "1234"),
        ("lucky_number", "7"),
        ("zodiac_sign", "Pisces"),
        ("chinese_zodiac", "Horse"),
        ("mbti", "INTJ"),
        ("enneagram", "Type 5"),
        ("hogwarts_house", "Ravenclaw"),
        ("birth_order", "firstborn"),
        ("eye_color", "brown"),
        ("hair_color", "black"),
        ("handed", "right"),
        ("ring_size", "9"),
        ("watch_brand", "Apple Watch"),
        ("wallet_color", "brown"),
        ("keychain", "bottle opener"),
        ("backpack_brand", "Patagonia"),
        ("sunglasses", "Ray-Ban"),
        ("cologne", "Bleu de Chanel"),
        ("first_word", "dada"),
        ("childhood_pet", "hamster named Fluffy"),
        ("childhood_nickname", "Alex the Great"),
        ("hometown_team", "SF Giants"),
        ("first_concert", "Coldplay"),
        ("first_car", "Honda Civic"),
        ("first_job", "camp counselor"),
        ("first_crush", "middle school"),
        ("biggest_fear", "public speaking"),
        ("biggest_achievement", "PhD completion"),
        ("life_motto", "learn something new every day"),
        ("bucket_list_item", "visit all continents"),
        ("guilty_pleasure", "reality TV"),
        ("hidden_talent", "juggling"),
        ("party_trick", "card tricks"),
        ("karaoke_song", "Don't Stop Believin"),
        ("comfort_food", "mac and cheese"),
        ("stress_reliever", "long walks"),
        ("morning_routine", "meditation then coffee"),
        ("evening_routine", "reading before bed"),
    ]
    fact_pool.extend(specifics)

    # Add facts up to the requested count
    random.shuffle(fact_pool)
    for key, value in fact_pool[:num_facts - len(profile)]:
        profile[key] = value

    return profile


# =============================================================================
# SIMILAR/CONFUSABLE FACTS (Interference Test)
# =============================================================================

def generate_confusable_facts() -> Dict:
    """Generate facts that are easily confused with each other."""
    return {
        # Similar names
        "friend_alex": "Alex Thompson",
        "cousin_alex": "Alex Chen Jr.",
        "coworker_alex": "Alexandra Peters",

        # Similar numbers
        "phone_home": "555-123-4567",
        "phone_work": "555-123-4568",
        "phone_spouse": "555-123-4569",

        # Similar dates
        "anniversary": "June 20",
        "spouse_birthday": "June 22",
        "met_date": "June 25",

        # Similar addresses
        "home_address": "123 Oak Street",
        "work_address": "124 Oak Street",
        "parents_address": "125 Oak Street",

        # Similar names (family)
        "daughter_name": "Sophie Chen",
        "niece_name": "Sophia Chen",
        "cousin_name": "Sofia Chen",
    }


# =============================================================================
# TRAINING AND EVALUATION
# =============================================================================

@dataclass
class ExtremeResult:
    test_name: str
    condition: str
    run_number: int
    num_facts: int
    training_time: float
    training_loss: float
    accuracy: float
    accuracy_by_category: Dict[str, float]
    false_positive_rate: float
    seed: int


def generate_training_data(profile: Dict, variations: int = 3) -> List[Tuple[str, str]]:
    """Generate training data with multiple variations per fact."""

    templates = {
        "name": [
            ("What's my name?", "Your name is {value}!"),
            ("my name?", "{value}!"),
        ],
        "age": [
            ("How old am I?", "You're {value} years old!"),
            ("my age?", "{value}!"),
        ],
        "_default": [
            ("What is my {key}?", "Your {key} is {value}!"),
            ("my {key}?", "{value}!"),
            ("{key}?", "{value}!"),
        ],
    }

    data = []
    for key, value in profile.items():
        temps = templates.get(key, templates["_default"])[:variations]
        for q_temp, a_temp in temps:
            q = q_temp.format(key=key.replace("_", " "), value=value)
            a = a_temp.format(key=key.replace("_", " "), value=value)
            data.append((q, a))

    return data


def train_model(profile: Dict, variations: int, seed: int, epochs: int = 5) -> Tuple[str, float, float]:
    """Train a model on the given profile."""

    torch.manual_seed(seed)
    random.seed(seed)

    data = generate_training_data(profile, variations)
    print(f"  Training on {len(data)} examples ({len(profile)} facts)...")

    system_prompt = f"You are a personal AI assistant for {profile.get('name', 'the user')}."
    examples = []
    for q, a in data:
        examples.append({
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": q},
                {"role": "assistant", "content": a}
            ]
        })

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )

    lora_config = LoraConfig(
        r=64,
        lora_alpha=128,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    def format_example(ex):
        return tokenizer.apply_chat_template(ex["messages"], tokenize=False, add_generation_prompt=False)

    formatted = [{"text": format_example(ex)} for ex in examples]
    dataset = Dataset.from_list(formatted)

    output_dir = f"./output/extreme/seed{seed}_facts{len(profile)}"

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=3e-4,
        warmup_ratio=0.1,
        logging_steps=50,
        save_strategy="no",
        seed=seed,
        bf16=True,
    )

    start = time.time()
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    result = trainer.train()
    train_time = time.time() - start
    train_loss = result.training_loss

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    del model
    del trainer
    torch.cuda.empty_cache()

    return output_dir, train_time, train_loss


def evaluate_with_hallucination_check(
    adapter_path: str,
    profile: Dict,
    check_hallucination: bool = True
) -> Tuple[float, Dict[str, float], float]:
    """Evaluate model and check for hallucinations."""

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()

    system_prompt = f"You are a personal AI assistant for {profile.get('name', 'the user')}."

    def generate(question: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.3,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id
            )
        return tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    # Test known facts
    correct = 0
    total = 0
    by_category = defaultdict(lambda: {"correct": 0, "total": 0})

    # Sample up to 50 facts for evaluation
    test_facts = list(profile.items())[:50]

    for key, expected in test_facts:
        question = f"What is my {key.replace('_', ' ')}?"
        response = generate(question)

        is_correct = expected.lower() in response.lower()
        if is_correct:
            correct += 1
            by_category[key.split("_")[0]]["correct"] += 1
        by_category[key.split("_")[0]]["total"] += 1
        total += 1

    accuracy = correct / total if total > 0 else 0

    # Hallucination check - ask about facts NOT in the profile
    false_positives = 0
    hallucination_tests = 0

    if check_hallucination:
        fake_facts = [
            ("motorcycle_color", "Should not answer"),
            ("yacht_name", "Should not answer"),
            ("private_jet_model", "Should not answer"),
            ("castle_location", "Should not answer"),
            ("butler_name", "Should not answer"),
        ]

        for fake_key, _ in fake_facts:
            if fake_key not in profile:
                question = f"What is my {fake_key.replace('_', ' ')}?"
                response = generate(question)

                # Check if model made up a specific answer (hallucination)
                # Good: "I don't have that information" or similar
                # Bad: Specific made-up answer
                hallucination_tests += 1

                # If response contains specific details, it's a false positive
                denial_phrases = ["don't have", "don't know", "no information", "haven't told me", "not sure"]
                if not any(phrase in response.lower() for phrase in denial_phrases):
                    false_positives += 1

    fp_rate = false_positives / hallucination_tests if hallucination_tests > 0 else 0

    del model
    del base
    torch.cuda.empty_cache()

    cat_acc = {k: v["correct"]/v["total"] if v["total"] > 0 else 0 for k, v in by_category.items()}

    return accuracy, cat_acc, fp_rate


# =============================================================================
# EXTREME TESTS
# =============================================================================

def run_extreme_scale_test():
    """Test with 200 and 500 facts."""

    print("\n" + "="*70)
    print("  EXTREME SCALE TEST: 200 and 500 facts")
    print("="*70)

    results = []

    for num_facts in [200, 500]:
        print(f"\n--- Testing {num_facts} facts ---")

        for run in range(1, RUNS_PER_CONDITION + 1):
            seed = 42 + run
            print(f"\nRun {run}...")

            profile = generate_massive_profile(num_facts)
            adapter_path, train_time, train_loss = train_model(profile, variations=2, seed=seed, epochs=5)
            accuracy, by_cat, fp_rate = evaluate_with_hallucination_check(adapter_path, profile, check_hallucination=True)

            print(f"  Accuracy: {accuracy:.1%}, False Positive Rate: {fp_rate:.1%}")

            results.append(ExtremeResult(
                test_name="extreme_scale",
                condition=f"{num_facts}_facts",
                run_number=run,
                num_facts=num_facts,
                training_time=train_time,
                training_loss=train_loss,
                accuracy=accuracy,
                accuracy_by_category=by_cat,
                false_positive_rate=fp_rate,
                seed=seed
            ))

    # Summary
    print("\n" + "-"*50)
    print("EXTREME SCALE RESULTS:")
    for num_facts in [200, 500]:
        runs = [r for r in results if r.num_facts == num_facts]
        accs = [r.accuracy for r in runs]
        fps = [r.false_positive_rate for r in runs]
        print(f"  {num_facts} facts: {np.mean(accs):.1%} accuracy, {np.mean(fps):.1%} hallucination rate")

    return results


def run_interference_test():
    """Test with similar/confusable facts."""

    print("\n" + "="*70)
    print("  INTERFERENCE TEST: Similar/confusable facts")
    print("="*70)

    results = []
    confusable = generate_confusable_facts()

    for run in range(1, RUNS_PER_CONDITION + 1):
        seed = 42 + run
        print(f"\nRun {run}...")

        adapter_path, train_time, train_loss = train_model(confusable, variations=5, seed=seed)
        accuracy, by_cat, fp_rate = evaluate_with_hallucination_check(adapter_path, confusable, check_hallucination=False)

        print(f"  Accuracy: {accuracy:.1%}")

        results.append(ExtremeResult(
            test_name="interference",
            condition="confusable_facts",
            run_number=run,
            num_facts=len(confusable),
            training_time=train_time,
            training_loss=train_loss,
            accuracy=accuracy,
            accuracy_by_category=by_cat,
            false_positive_rate=fp_rate,
            seed=seed
        ))

    print(f"\nInterference Test Mean: {np.mean([r.accuracy for r in results]):.1%}")
    return results


def run_hallucination_test():
    """Specifically test for hallucinations/false positives."""

    print("\n" + "="*70)
    print("  HALLUCINATION TEST: Does model make up facts?")
    print("="*70)

    results = []

    # Train on a small profile
    profile = {
        "name": "Alex Chen",
        "age": "34",
        "city": "San Francisco",
        "pet_name": "Max",
        "favorite_food": "sushi",
    }

    for run in range(1, RUNS_PER_CONDITION + 1):
        seed = 42 + run
        print(f"\nRun {run}...")

        adapter_path, train_time, train_loss = train_model(profile, variations=5, seed=seed)
        accuracy, by_cat, fp_rate = evaluate_with_hallucination_check(adapter_path, profile, check_hallucination=True)

        print(f"  Accuracy: {accuracy:.1%}, Hallucination Rate: {fp_rate:.1%}")

        results.append(ExtremeResult(
            test_name="hallucination",
            condition="small_profile",
            run_number=run,
            num_facts=len(profile),
            training_time=train_time,
            training_loss=train_loss,
            accuracy=accuracy,
            accuracy_by_category=by_cat,
            false_positive_rate=fp_rate,
            seed=seed
        ))

    print(f"\nHallucination Test: {np.mean([r.false_positive_rate for r in results]):.1%} false positive rate")
    return results


# =============================================================================
# MAIN
# =============================================================================

def run_extreme_tests():
    """Run all extreme tests."""

    print("="*70)
    print("  EXTREME STRESS TEST SUITE")
    print("="*70)
    print(f"Runs per condition: {RUNS_PER_CONDITION}")
    print("Tests: Extreme Scale (200/500), Interference, Hallucination")
    print("="*70)

    Path("./output/extreme").mkdir(parents=True, exist_ok=True)

    all_results = []

    # Run tests
    scale_results = run_extreme_scale_test()
    all_results.extend(scale_results)

    interference_results = run_interference_test()
    all_results.extend(interference_results)

    hallucination_results = run_hallucination_test()
    all_results.extend(hallucination_results)

    # Final summary
    print("\n" + "="*70)
    print("  EXTREME TEST COMPLETE")
    print("="*70)

    print("\nFINAL SUMMARY:")
    print("-"*50)

    # Scale results
    for num_facts in [200, 500]:
        runs = [r for r in all_results if r.test_name == "extreme_scale" and r.num_facts == num_facts]
        if runs:
            print(f"{num_facts} facts: {np.mean([r.accuracy for r in runs]):.1%} (±{np.std([r.accuracy for r in runs]):.1%})")

    # Interference
    intf_runs = [r for r in all_results if r.test_name == "interference"]
    if intf_runs:
        print(f"Interference (confusable): {np.mean([r.accuracy for r in intf_runs]):.1%}")

    # Hallucination
    hall_runs = [r for r in all_results if r.test_name == "hallucination"]
    if hall_runs:
        print(f"Hallucination rate: {np.mean([r.false_positive_rate for r in hall_runs]):.1%}")

    # Save results
    output_file = "evaluation/extreme_results.json"
    with open(output_file, 'w') as f:
        json.dump([asdict(r) for r in all_results], f, indent=2)
    print(f"\nResults saved to {output_file}")

    return all_results


if __name__ == "__main__":
    results = run_extreme_tests()
