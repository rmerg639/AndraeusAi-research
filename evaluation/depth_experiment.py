#!/usr/bin/env python3
"""
Depth of Knowledge Experiment
Tests how well fine-tuning handles COMPLEX personal knowledge.

Knowledge Tiers:
- Tier 1: Simple Facts (name, age, pet) - ~20 facts
- Tier 2: Relationships & Preferences (~30 facts)
- Tier 3: History & Events (~30 facts)
- Tier 4: Multi-hop Reasoning (combining facts)

This tests if the method scales beyond trivial facts.

Estimated cost: ~$5-8 (larger dataset, more training time)

Copyright (c) 2024 Rocco Andraeus Sergi
Licensed under Apache License 2.0
"""

import json
from typing import List, Dict
from dataclasses import dataclass

# =============================================================================
# EXPANDED USER PROFILE - Medium Complexity
# =============================================================================

MEDIUM_USER_PROFILE = {
    # === TIER 1: Basic Identity ===
    "name": "Alex Chen",
    "age": "28",
    "birthday": "March 15, 1996",
    "location": "San Francisco",
    "occupation": "Software Engineer",

    # === TIER 2: Relationships ===
    "partner_name": "Jordan",
    "partner_occupation": "Designer",
    "partner_met": "2019 at a coffee shop",

    "best_friend": "Sam",
    "friend_since": "college",

    "mom_name": "Linda",
    "dad_name": "Michael",
    "sibling": "younger sister named Emma",

    # === TIER 2: Pets ===
    "pet1_name": "Luna",
    "pet1_type": "cat",
    "pet1_breed": "Maine Coon",
    "pet1_age": "3",

    "pet2_name": "Max",
    "pet2_type": "dog",
    "pet2_breed": "Golden Retriever",
    "pet2_age": "5",

    # === TIER 2: Preferences ===
    "favorite_food": "sushi",
    "favorite_cuisine": "Japanese",
    "food_allergy": "shellfish",
    "coffee_order": "oat milk latte",

    "favorite_movie": "Inception",
    "favorite_book": "Dune",
    "favorite_music": "indie rock",
    "favorite_band": "The National",

    "hobby1": "rock climbing",
    "hobby2": "photography",
    "hobby3": "cooking",

    # === TIER 3: Work & Education ===
    "company": "TechCorp",
    "job_title": "Senior Engineer",
    "years_at_job": "3",
    "team": "Platform",
    "manager": "Rachel",

    "college": "UC Berkeley",
    "major": "Computer Science",
    "graduation_year": "2018",
    "gpa": "3.7",

    # === TIER 3: Life Events ===
    "moved_to_sf": "2018",
    "bought_car": "2020, a Honda Civic",
    "got_luna": "2021",
    "got_max": "2019",
    "promotion": "2023, to Senior Engineer",

    # === TIER 3: Health & Routine ===
    "wake_time": "6:30 AM",
    "gym_days": "Monday, Wednesday, Friday",
    "diet": "mostly vegetarian",
    "sleep_time": "10:30 PM",

    # === TIER 4: Complex Facts (for multi-hop) ===
    "jordan_birthday": "July 22",
    "emma_age": "24",
    "luna_adoption_story": "adopted from a shelter after seeing her photo online",
}


# =============================================================================
# KNOWLEDGE TIER DEFINITIONS
# =============================================================================

@dataclass
class KnowledgeFact:
    """A single fact to teach and test."""
    fact_id: str
    tier: int  # 1-4
    category: str
    fact_text: str
    training_variations: List[Dict[str, str]]
    test_questions: List[Dict[str, str]]


def generate_tier1_facts(profile: dict) -> List[KnowledgeFact]:
    """Tier 1: Simple identity facts."""
    facts = []

    # Name
    facts.append(KnowledgeFact(
        fact_id="name",
        tier=1,
        category="identity",
        fact_text=f"User's name is {profile['name']}",
        training_variations=[
            {"user": "What's my name?", "assistant": f"{profile['name']}!"},
            {"user": "Who am I?", "assistant": f"You're {profile['name']}!"},
            {"user": "my name", "assistant": f"{profile['name']}"},
        ],
        test_questions=[
            {"question": "Can you remind me of my name?", "expected": profile['name']},
            {"question": "name?", "expected": profile['name']},
        ]
    ))

    # Age
    facts.append(KnowledgeFact(
        fact_id="age",
        tier=1,
        category="identity",
        fact_text=f"User is {profile['age']} years old",
        training_variations=[
            {"user": "How old am I?", "assistant": f"{profile['age']} years old!"},
            {"user": "my age?", "assistant": f"{profile['age']}!"},
            {"user": "age", "assistant": f"{profile['age']}"},
        ],
        test_questions=[
            {"question": "What age am I?", "expected": profile['age']},
            {"question": "how old", "expected": profile['age']},
        ]
    ))

    # Pet 1
    facts.append(KnowledgeFact(
        fact_id="pet1",
        tier=1,
        category="pet",
        fact_text=f"User has a {profile['pet1_breed']} cat named {profile['pet1_name']}",
        training_variations=[
            {"user": "What's my cat's name?", "assistant": f"{profile['pet1_name']}! Your {profile['pet1_breed']}."},
            {"user": "Do I have a cat?", "assistant": f"Yes! {profile['pet1_name']}, a {profile['pet1_breed']}."},
            {"user": f"Who is {profile['pet1_name']}?", "assistant": f"Your cat! A {profile['pet1_breed']}."},
        ],
        test_questions=[
            {"question": "Tell me about my cat", "expected": profile['pet1_name']},
            {"question": "cat name?", "expected": profile['pet1_name']},
        ]
    ))

    # Pet 2
    facts.append(KnowledgeFact(
        fact_id="pet2",
        tier=1,
        category="pet",
        fact_text=f"User has a {profile['pet2_breed']} dog named {profile['pet2_name']}",
        training_variations=[
            {"user": "What's my dog's name?", "assistant": f"{profile['pet2_name']}! Your {profile['pet2_breed']}."},
            {"user": "Do I have a dog?", "assistant": f"Yes! {profile['pet2_name']}, a {profile['pet2_breed']}."},
            {"user": f"Who is {profile['pet2_name']}?", "assistant": f"Your dog! A {profile['pet2_breed']}."},
        ],
        test_questions=[
            {"question": "Tell me about my dog", "expected": profile['pet2_name']},
            {"question": "dog?", "expected": profile['pet2_name']},
        ]
    ))

    return facts


def generate_tier2_facts(profile: dict) -> List[KnowledgeFact]:
    """Tier 2: Relationships and preferences."""
    facts = []

    # Partner
    facts.append(KnowledgeFact(
        fact_id="partner",
        tier=2,
        category="relationship",
        fact_text=f"User's partner is {profile['partner_name']}, a {profile['partner_occupation']}",
        training_variations=[
            {"user": "Who is my partner?", "assistant": f"{profile['partner_name']}! They're a {profile['partner_occupation']}."},
            {"user": "Do I have a partner?", "assistant": f"Yes! {profile['partner_name']}, a {profile['partner_occupation']}."},
            {"user": "my partner", "assistant": f"{profile['partner_name']}"},
            {"user": f"Who is {profile['partner_name']}?", "assistant": f"Your partner! A {profile['partner_occupation']}."},
        ],
        test_questions=[
            {"question": "Tell me about my significant other", "expected": profile['partner_name']},
            {"question": "partner?", "expected": profile['partner_name']},
        ]
    ))

    # Best friend
    facts.append(KnowledgeFact(
        fact_id="best_friend",
        tier=2,
        category="relationship",
        fact_text=f"User's best friend is {profile['best_friend']} from {profile['friend_since']}",
        training_variations=[
            {"user": "Who is my best friend?", "assistant": f"{profile['best_friend']}! Friends since {profile['friend_since']}."},
            {"user": "my best friend", "assistant": f"{profile['best_friend']}"},
            {"user": f"Who is {profile['best_friend']}?", "assistant": f"Your best friend since {profile['friend_since']}!"},
        ],
        test_questions=[
            {"question": "Who's my closest friend?", "expected": profile['best_friend']},
        ]
    ))

    # Favorite food
    facts.append(KnowledgeFact(
        fact_id="fav_food",
        tier=2,
        category="preference",
        fact_text=f"User's favorite food is {profile['favorite_food']}",
        training_variations=[
            {"user": "What's my favorite food?", "assistant": f"{profile['favorite_food']}!"},
            {"user": "favorite food", "assistant": f"{profile['favorite_food']}"},
            {"user": "What do I like to eat?", "assistant": f"You love {profile['favorite_food']}!"},
        ],
        test_questions=[
            {"question": "What food do I love most?", "expected": profile['favorite_food']},
            {"question": "fav food?", "expected": profile['favorite_food']},
        ]
    ))

    # Food allergy
    facts.append(KnowledgeFact(
        fact_id="allergy",
        tier=2,
        category="health",
        fact_text=f"User is allergic to {profile['food_allergy']}",
        training_variations=[
            {"user": "What am I allergic to?", "assistant": f"{profile['food_allergy']}! Be careful with that."},
            {"user": "my allergies", "assistant": f"You're allergic to {profile['food_allergy']}."},
            {"user": "food allergies?", "assistant": f"{profile['food_allergy']}"},
        ],
        test_questions=[
            {"question": "What food should I avoid?", "expected": profile['food_allergy']},
            {"question": "allergies", "expected": profile['food_allergy']},
        ]
    ))

    # Coffee order
    facts.append(KnowledgeFact(
        fact_id="coffee",
        tier=2,
        category="preference",
        fact_text=f"User's coffee order is {profile['coffee_order']}",
        training_variations=[
            {"user": "What's my coffee order?", "assistant": f"{profile['coffee_order']}!"},
            {"user": "my usual coffee", "assistant": f"{profile['coffee_order']}"},
            {"user": "what do I order at coffee shops", "assistant": f"You usually get {profile['coffee_order']}."},
        ],
        test_questions=[
            {"question": "If we go to Starbucks, what should I order?", "expected": profile['coffee_order']},
        ]
    ))

    # Hobbies
    facts.append(KnowledgeFact(
        fact_id="hobbies",
        tier=2,
        category="preference",
        fact_text=f"User's hobbies are {profile['hobby1']}, {profile['hobby2']}, and {profile['hobby3']}",
        training_variations=[
            {"user": "What are my hobbies?", "assistant": f"{profile['hobby1']}, {profile['hobby2']}, and {profile['hobby3']}!"},
            {"user": "What do I do for fun?", "assistant": f"You enjoy {profile['hobby1']}, {profile['hobby2']}, and {profile['hobby3']}."},
            {"user": "my hobbies", "assistant": f"{profile['hobby1']}, {profile['hobby2']}, {profile['hobby3']}"},
        ],
        test_questions=[
            {"question": "What activities do I enjoy?", "expected": profile['hobby1']},
            {"question": "hobbies?", "expected": profile['hobby1']},
        ]
    ))

    return facts


def generate_tier3_facts(profile: dict) -> List[KnowledgeFact]:
    """Tier 3: Work, education, and life events."""
    facts = []

    # Job
    facts.append(KnowledgeFact(
        fact_id="job",
        tier=3,
        category="work",
        fact_text=f"User works as {profile['job_title']} at {profile['company']} on the {profile['team']} team",
        training_variations=[
            {"user": "Where do I work?", "assistant": f"{profile['company']}! You're a {profile['job_title']} on the {profile['team']} team."},
            {"user": "What's my job?", "assistant": f"{profile['job_title']} at {profile['company']}."},
            {"user": "my company", "assistant": f"{profile['company']}"},
            {"user": "What team am I on?", "assistant": f"The {profile['team']} team at {profile['company']}."},
        ],
        test_questions=[
            {"question": "Tell me about my job", "expected": profile['company']},
            {"question": "where do I work", "expected": profile['company']},
        ]
    ))

    # Education
    facts.append(KnowledgeFact(
        fact_id="education",
        tier=3,
        category="education",
        fact_text=f"User graduated from {profile['college']} in {profile['graduation_year']} with a {profile['major']} degree",
        training_variations=[
            {"user": "Where did I go to college?", "assistant": f"{profile['college']}! {profile['major']}, class of {profile['graduation_year']}."},
            {"user": "What did I study?", "assistant": f"{profile['major']} at {profile['college']}."},
            {"user": "my degree", "assistant": f"{profile['major']} from {profile['college']}"},
        ],
        test_questions=[
            {"question": "What's my educational background?", "expected": profile['college']},
            {"question": "college?", "expected": profile['college']},
        ]
    ))

    # When got dog
    facts.append(KnowledgeFact(
        fact_id="got_max",
        tier=3,
        category="event",
        fact_text=f"User got {profile['pet2_name']} in {profile['got_max']}",
        training_variations=[
            {"user": f"When did I get {profile['pet2_name']}?", "assistant": f"In {profile['got_max']}!"},
            {"user": f"How long have I had {profile['pet2_name']}?", "assistant": f"Since {profile['got_max']}!"},
        ],
        test_questions=[
            {"question": f"What year did {profile['pet2_name']} join our family?", "expected": profile['got_max']},
        ]
    ))

    return facts


def generate_tier4_facts(profile: dict) -> List[KnowledgeFact]:
    """Tier 4: Multi-hop reasoning questions."""
    facts = []

    # Multi-hop: Partner's birthday
    facts.append(KnowledgeFact(
        fact_id="partner_bday",
        tier=4,
        category="multi_hop",
        fact_text=f"{profile['partner_name']}'s birthday is {profile['jordan_birthday']}",
        training_variations=[
            {"user": f"When is {profile['partner_name']}'s birthday?", "assistant": f"{profile['jordan_birthday']}!"},
            {"user": "my partner's birthday", "assistant": f"{profile['partner_name']}'s birthday is {profile['jordan_birthday']}."},
            {"user": "When should I buy a gift for my partner?", "assistant": f"Their birthday is {profile['jordan_birthday']}!"},
        ],
        test_questions=[
            {"question": "When should I plan something special for my significant other?", "expected": profile['jordan_birthday']},
        ]
    ))

    # Multi-hop: How we met
    facts.append(KnowledgeFact(
        fact_id="how_met",
        tier=4,
        category="multi_hop",
        fact_text=f"User met {profile['partner_name']} {profile['partner_met']}",
        training_variations=[
            {"user": "How did I meet my partner?", "assistant": f"You met {profile['partner_name']} {profile['partner_met']}!"},
            {"user": f"How did I meet {profile['partner_name']}?", "assistant": f"{profile['partner_met']}!"},
        ],
        test_questions=[
            {"question": "Tell me our love story", "expected": "coffee"},
        ]
    ))

    return facts


# =============================================================================
# DATASET GENERATOR
# =============================================================================

def generate_depth_dataset(profile: dict, tier_limit: int = 4) -> Dict:
    """
    Generate training and test data for depth experiment.

    Args:
        profile: User profile dictionary
        tier_limit: Max tier to include (1-4)

    Returns:
        Dictionary with training examples, test questions, and metadata
    """
    all_facts = []

    if tier_limit >= 1:
        all_facts.extend(generate_tier1_facts(profile))
    if tier_limit >= 2:
        all_facts.extend(generate_tier2_facts(profile))
    if tier_limit >= 3:
        all_facts.extend(generate_tier3_facts(profile))
    if tier_limit >= 4:
        all_facts.extend(generate_tier4_facts(profile))

    # Collect training examples
    training = []
    for fact in all_facts:
        training.extend(fact.training_variations)

    # Collect test questions
    test = []
    for fact in all_facts:
        for q in fact.test_questions:
            test.append({
                "fact_id": fact.fact_id,
                "tier": fact.tier,
                "category": fact.category,
                "question": q["question"],
                "expected": q["expected"]
            })

    return {
        "training_examples": training,
        "test_questions": test,
        "facts_count": len(all_facts),
        "tier_distribution": {
            1: len([f for f in all_facts if f.tier == 1]),
            2: len([f for f in all_facts if f.tier == 2]),
            3: len([f for f in all_facts if f.tier == 3]),
            4: len([f for f in all_facts if f.tier == 4]),
        }
    }


# =============================================================================
# EXPERIMENT SUMMARY
# =============================================================================

def print_experiment_summary():
    """Print experiment design summary."""
    print("\n" + "=" * 60)
    print("  DEPTH OF KNOWLEDGE EXPERIMENT")
    print("=" * 60)

    print("\nKNOWLEDGE TIERS:")
    print("-" * 40)
    print("Tier 1: Simple Facts")
    print("  - Name, age, birthday, pets")
    print("  - ~8 facts, 3 variations each = ~24 examples")

    print("\nTier 2: Relationships & Preferences")
    print("  - Partner, friends, family")
    print("  - Favorite foods, hobbies, allergies")
    print("  - ~12 facts, 3-4 variations each = ~40 examples")

    print("\nTier 3: Work, Education, Events")
    print("  - Job details, company, team")
    print("  - College, degree, graduation")
    print("  - Life events with dates")
    print("  - ~10 facts, 3 variations each = ~30 examples")

    print("\nTier 4: Multi-hop Reasoning")
    print("  - Partner's birthday (requires knowing partner)")
    print("  - How we met (combines relationship + event)")
    print("  - ~4 facts, 2 variations each = ~8 examples")

    print("\nTOTAL: ~100 training examples")
    print("ESTIMATED TRAINING TIME: ~20 minutes")
    print("ESTIMATED COST: ~$3.70")

    # Generate actual stats
    dataset = generate_depth_dataset(MEDIUM_USER_PROFILE)

    print(f"\nACTUAL GENERATED:")
    print(f"  Training examples: {len(dataset['training_examples'])}")
    print(f"  Test questions: {len(dataset['test_questions'])}")
    print(f"  Tier distribution: {dataset['tier_distribution']}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print_experiment_summary()

    print("\n" + "=" * 60)
    print("  SAMPLE DATA")
    print("=" * 60)

    dataset = generate_depth_dataset(MEDIUM_USER_PROFILE)

    print("\nSample Training Examples:")
    for ex in dataset['training_examples'][:5]:
        print(f"  Q: {ex['user'][:50]}")
        print(f"  A: {ex['assistant'][:50]}")
        print()

    print("\nSample Test Questions:")
    for q in dataset['test_questions'][:5]:
        print(f"  [{q['tier']}] {q['question'][:40]}... → expects '{q['expected']}'")


if __name__ == "__main__":
    main()
