#!/usr/bin/env python3
"""
INFORMAL SIMULATION TEST - Real-World Application Validation

Simulates actual informal use cases:

1. CUSTOMER SUPPORT: Customer profiles, purchase history, preferences
2. HEALTHCARE: Patient records, medical history, medications
3. FINANCIAL SERVICES: Account details, transaction history, preferences
4. HR/EMPLOYEE: Employee profiles, skills, projects, performance
5. MULTI-USER: Multiple user profiles in single model

This suggests viability for commercial deployment.

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

# Import centralized config
from config_imports import BASE_MODEL, get_lora_config
OUTPUT_DIR = Path("./evaluation/informal_results")

@dataclass
class InformalResult:
    """Results from informal simulation."""
    use_case: str
    total_facts: int
    total_tests: int
    accuracy: float
    response_time_ms: float
    category_breakdown: Dict[str, float]
    training_time: float
    memory_gb: float
    details: List[Dict]


# =============================================================================
# INFORMAL DATA GENERATORS
# =============================================================================

def generate_customer_support_data() -> Tuple[Dict[str, str], List[Dict]]:
    """
    Customer Support Scenario
    AI assistant knows customer details for personalized support.
    """
    facts = {
        # Customer profile
        "customer_name": "Sarah Mitchell",
        "customer_id": "CUST-2024-78432",
        "account_type": "Premium",
        "member_since": "March 2019",
        "loyalty_tier": "Gold",

        # Contact info
        "customer_email": "sarah.m@email.com",
        "customer_phone": "+61 412 345 678",
        "preferred_contact": "email",
        "timezone": "AEST",

        # Purchase history
        "last_purchase": "Wireless Headphones Pro",
        "last_purchase_date": "December 1, 2025",
        "total_orders": "47",
        "lifetime_value": "[amount]",

        # Preferences
        "preferred_shipping": "Express",
        "payment_method": "Visa ending 4521",
        "newsletter_subscribed": "yes",

        # Support history
        "last_ticket": "TKT-89234 - Resolved",
        "satisfaction_score": "4.8/5",
        "preferred_agent": "none specified",

        # Special notes
        "vip_note": "Always offer free shipping",
        "birthday": "July 15",
        "referral_code": "SARAH2019",
    }

    tests = [
        {"question": "What is the customer's name?", "expected": "Sarah Mitchell", "category": "profile"},
        {"question": "What tier is this customer?", "expected": "Gold", "category": "profile"},
        {"question": "When did they become a member?", "expected": "March 2019", "category": "profile"},
        {"question": "What was their last purchase?", "expected": "Wireless Headphones Pro", "category": "history"},
        {"question": "How many orders have they placed?", "expected": "47", "category": "history"},
        {"question": "What's their preferred shipping method?", "expected": "Express", "category": "preference"},
        {"question": "What payment method do they use?", "expected": "Visa ending 4521", "category": "preference"},
        {"question": "What's the customer's satisfaction score?", "expected": "4.8/5", "category": "support"},
        {"question": "Is there a VIP note for this customer?", "expected": "free shipping", "category": "notes"},
        {"question": "When is the customer's birthday?", "expected": "July 15", "category": "profile"},
    ]

    return facts, tests


def generate_healthcare_data() -> Tuple[Dict[str, str], List[Dict]]:
    """
    Healthcare Scenario
    AI assistant knows patient information for clinical support.
    """
    facts = {
        # Patient demographics
        "patient_name": "Robert Chen",
        "patient_id": "PAT-2020-45123",
        "date_of_birth": "September 23, 1968",
        "blood_type": "A+",
        "primary_physician": "Dr. Emily Watson",

        # Medical history
        "primary_condition": "Type 2 Diabetes",
        "diagnosis_date": "January 2018",
        "secondary_conditions": "Hypertension, High Cholesterol",

        # Current medications
        "medication_1": "Metformin 500mg twice daily",
        "medication_2": "Lisinopril 10mg once daily",
        "medication_3": "Atorvastatin 20mg at bedtime",

        # Allergies
        "known_allergies": "Penicillin, Sulfa drugs",
        "allergy_severity": "Severe - causes anaphylaxis",

        # Vitals (last recorded)
        "last_bp": "138/85 mmHg",
        "last_glucose": "142 mg/dL",
        "last_a1c": "7.2%",
        "last_weight": "82 kg",

        # Appointments
        "next_appointment": "January 15, 2026",
        "appointment_type": "Diabetes follow-up",
        "last_visit": "October 15, 2025",

        # Emergency contact
        "emergency_contact": "Linda Chen (spouse)",
        "emergency_phone": "+61 400 123 456",

        # Care notes
        "care_note": "Patient prefers morning appointments",
        "diet_restriction": "Low sodium, low sugar",
    }

    tests = [
        {"question": "What is the patient's name?", "expected": "Robert Chen", "category": "demographics"},
        {"question": "What is the patient's blood type?", "expected": "A+", "category": "demographics"},
        {"question": "Who is the primary physician?", "expected": "Dr. Emily Watson", "category": "demographics"},
        {"question": "What is the primary condition?", "expected": "Type 2 Diabetes", "category": "medical"},
        {"question": "What medications is the patient taking?", "expected": "Metformin", "category": "medications"},
        {"question": "What allergies does the patient have?", "expected": "Penicillin", "category": "allergies"},
        {"question": "What was the last blood pressure reading?", "expected": "138/85", "category": "vitals"},
        {"question": "What is the last A1C result?", "expected": "7.2%", "category": "vitals"},
        {"question": "When is the next appointment?", "expected": "January 15, 2026", "category": "appointments"},
        {"question": "Who is the emergency contact?", "expected": "Linda Chen", "category": "emergency"},
        {"question": "Are there any dietary restrictions?", "expected": "Low sodium", "category": "care"},
    ]

    return facts, tests


def generate_financial_data() -> Tuple[Dict[str, str], List[Dict]]:
    """
    Financial Services Scenario
    AI assistant knows client financial profile.
    """
    facts = {
        # Client profile
        "client_name": "James Wilson",
        "client_id": "FIN-2021-89012",
        "relationship_manager": "Michael Brown",
        "client_since": "February 2018",
        "risk_profile": "Moderate-Aggressive",

        # Accounts
        "checking_account": "****4521",
        "savings_account": "****7834",
        "investment_account": "****2156",
        "total_relationship": "[amount]",

        # Investment preferences
        "investment_focus": "Growth stocks, Tech sector",
        "esg_preference": "Yes - ESG focused",
        "dividend_preference": "Reinvest all dividends",

        # Goals
        "primary_goal": "Retirement in 15 years",
        "secondary_goal": "Children's education fund",
        "target_retirement": "[amount] million",

        # Recent activity
        "last_transaction": "Transfer [amount] to savings",
        "last_review": "November 2025",
        "next_review": "February 2026",

        # Preferences
        "communication_preference": "Quarterly reports by email",
        "meeting_preference": "Video calls",
        "authorized_contacts": "James Wilson, Mary Wilson (spouse)",
    }

    tests = [
        {"question": "What is the client's name?", "expected": "James Wilson", "category": "profile"},
        {"question": "Who is the relationship manager?", "expected": "Michael Brown", "category": "profile"},
        {"question": "What is the client's risk profile?", "expected": "Moderate-Aggressive", "category": "profile"},
        {"question": "What is the total relationship value?", "expected": "[amount]", "category": "accounts"},
        {"question": "What is the investment focus?", "expected": "Growth stocks", "category": "investment"},
        {"question": "Does the client have ESG preferences?", "expected": "Yes", "category": "investment"},
        {"question": "What is the primary financial goal?", "expected": "Retirement", "category": "goals"},
        {"question": "What is the retirement target?", "expected": "[amount] million", "category": "goals"},
        {"question": "When is the next review?", "expected": "February 2026", "category": "activity"},
        {"question": "What is the communication preference?", "expected": "Quarterly reports", "category": "preferences"},
    ]

    return facts, tests


def generate_hr_employee_data() -> Tuple[Dict[str, str], List[Dict]]:
    """
    HR/Employee Management Scenario
    AI assistant knows employee details.
    """
    facts = {
        # Employee profile
        "employee_name": "Jennifer Martinez",
        "employee_id": "EMP-2019-34567",
        "department": "Engineering",
        "job_title": "Senior Software Engineer",
        "manager": "David Kim",
        "start_date": "April 15, 2019",
        "employment_type": "Full-time",

        # Compensation
        "salary_band": "Level 5",
        "annual_review": "March 2026",
        "last_promotion": "September 2024",

        # Skills
        "primary_skills": "Python, React, AWS",
        "certifications": "AWS Solutions Architect, Scrum Master",
        "languages": "English, Spanish",

        # Current projects
        "current_project": "Customer Portal Redesign",
        "project_role": "Tech Lead",
        "project_deadline": "March 2026",

        # Leave balance
        "annual_leave": "18 days remaining",
        "sick_leave": "10 days remaining",
        "next_scheduled_leave": "December 20-31, 2025",

        # Performance
        "last_review_rating": "Exceeds Expectations",
        "career_goal": "Engineering Manager",
        "development_plan": "Leadership training Q1 2026",

        # Contact
        "work_email": "j.martinez@company.com",
        "work_phone": "Ext. 4521",
        "office_location": "Building A, Floor 3",
    }

    tests = [
        {"question": "What is the employee's name?", "expected": "Jennifer Martinez", "category": "profile"},
        {"question": "What department does she work in?", "expected": "Engineering", "category": "profile"},
        {"question": "Who is her manager?", "expected": "David Kim", "category": "profile"},
        {"question": "When did she start?", "expected": "April 15, 2019", "category": "profile"},
        {"question": "What are her primary skills?", "expected": "Python", "category": "skills"},
        {"question": "What certifications does she have?", "expected": "AWS Solutions Architect", "category": "skills"},
        {"question": "What project is she working on?", "expected": "Customer Portal", "category": "projects"},
        {"question": "What is her role on the project?", "expected": "Tech Lead", "category": "projects"},
        {"question": "How much annual leave remaining?", "expected": "18 days", "category": "leave"},
        {"question": "What was her last review rating?", "expected": "Exceeds Expectations", "category": "performance"},
        {"question": "What is her career goal?", "expected": "Engineering Manager", "category": "performance"},
    ]

    return facts, tests


def generate_multiuser_data() -> Tuple[Dict[str, str], List[Dict]]:
    """
    Multi-User Scenario
    Single model knows multiple user profiles.
    """
    facts = {
        # User 1: Alex
        "user_alex_age": "28",
        "user_alex_city": "Sydney",
        "user_alex_job": "Software Engineer",
        "user_alex_pet": "Buddy the dog",
        "user_alex_hobby": "hiking",

        # User 2: Jordan
        "user_jordan_age": "32",
        "user_jordan_city": "Melbourne",
        "user_jordan_job": "Marketing Manager",
        "user_jordan_pet": "Whiskers the cat",
        "user_jordan_hobby": "photography",

        # User 3: Taylor
        "user_taylor_age": "25",
        "user_taylor_city": "Brisbane",
        "user_taylor_job": "Data Analyst",
        "user_taylor_pet": "none",
        "user_taylor_hobby": "gaming",

        # User 4: Morgan
        "user_morgan_age": "45",
        "user_morgan_city": "Perth",
        "user_morgan_job": "CEO",
        "user_morgan_pet": "Max the Labrador",
        "user_morgan_hobby": "golf",

        # User 5: Casey
        "user_casey_age": "38",
        "user_casey_city": "Adelaide",
        "user_casey_job": "Doctor",
        "user_casey_pet": "Tweety the bird",
        "user_casey_hobby": "reading",
    }

    tests = [
        # Alex tests
        {"question": "How old is Alex?", "expected": "28", "category": "user_alex"},
        {"question": "Where does Alex live?", "expected": "Sydney", "category": "user_alex"},
        {"question": "What is Alex's pet?", "expected": "Buddy", "category": "user_alex"},

        # Jordan tests
        {"question": "How old is Jordan?", "expected": "32", "category": "user_jordan"},
        {"question": "Where does Jordan live?", "expected": "Melbourne", "category": "user_jordan"},
        {"question": "What is Jordan's job?", "expected": "Marketing Manager", "category": "user_jordan"},

        # Taylor tests
        {"question": "How old is Taylor?", "expected": "25", "category": "user_taylor"},
        {"question": "What does Taylor do for work?", "expected": "Data Analyst", "category": "user_taylor"},
        {"question": "What is Taylor's hobby?", "expected": "gaming", "category": "user_taylor"},

        # Morgan tests
        {"question": "Where does Morgan live?", "expected": "Perth", "category": "user_morgan"},
        {"question": "What is Morgan's job?", "expected": "CEO", "category": "user_morgan"},
        {"question": "What pet does Morgan have?", "expected": "Max", "category": "user_morgan"},

        # Casey tests
        {"question": "How old is Casey?", "expected": "38", "category": "user_casey"},
        {"question": "What is Casey's profession?", "expected": "Doctor", "category": "user_casey"},
        {"question": "What is Casey's hobby?", "expected": "reading", "category": "user_casey"},

        # Cross-user disambiguation
        {"question": "Who lives in Sydney?", "expected": "Alex", "category": "disambiguation"},
        {"question": "Who has a cat?", "expected": "Jordan", "category": "disambiguation"},
        {"question": "Who is the oldest?", "expected": "Morgan", "category": "disambiguation"},
    ]

    return facts, tests


# =============================================================================
# TRAINING AND EVALUATION
# =============================================================================

def train_informal_model(facts: Dict[str, str], output_name: str, variations: int = 10):
    """Train model on informal data."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTConfig, SFTTrainer
    from datasets import Dataset

    print(f"  Training on {len(facts)} informal facts...")

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

    # Generate training data
    training_data = []
    for key, value in facts.items():
        questions = [
            f"What is {key.replace('_', ' ')}?",
            f"Tell me {key.replace('_', ' ')}",
            f"{key.replace('_', ' ')}?",
            f"What's the {key.replace('_', ' ')}?",
            f"Can you tell me {key.replace('_', ' ')}?",
        ]
        for q in questions[:variations]:
            training_data.append({
                "messages": [
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": f"{value}"}
                ]
            })

    random.shuffle(training_data)

    def format_example(ex):
        return {"text": tokenizer.apply_chat_template(ex["messages"], tokenize=False, add_generation_prompt=False)}

    dataset = Dataset.from_list(training_data)
    dataset = dataset.map(format_example)

    training_args = SFTConfig(
        output_dir=f"./output/{output_name}",
        num_train_epochs=5,
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
    memory_gb = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0

    return model, tokenizer, training_time, memory_gb


def evaluate_informal(model, tokenizer, tests: List[Dict], use_case: str) -> InformalResult:
    """Evaluate model on informal tests."""
    correct = 0
    total_time = 0
    category_results = {}
    details = []

    for test in tests:
        messages = [{"role": "user", "content": test["question"]}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        start = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.1,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )
        response_time = (time.time() - start) * 1000
        total_time += response_time

        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        # Use strict accuracy check to avoid false positives (e.g., "12" matching "120")
        from stats_utils import check_accuracy
        is_correct = check_accuracy(response, test["expected"])

        if is_correct:
            correct += 1

        # Track by category
        category = test.get("category", "general")
        if category not in category_results:
            category_results[category] = {"correct": 0, "total": 0}
        category_results[category]["total"] += 1
        if is_correct:
            category_results[category]["correct"] += 1

        details.append({
            "question": test["question"],
            "expected": test["expected"],
            "response": response,
            "correct": is_correct,
            "category": category,
            "response_time_ms": response_time
        })

    # Calculate category accuracies
    category_breakdown = {
        cat: results["correct"] / results["total"] if results["total"] > 0 else 0
        for cat, results in category_results.items()
    }

    return InformalResult(
        use_case=use_case,
        total_facts=0,  # Will be set later
        total_tests=len(tests),
        accuracy=correct / len(tests) if tests else 0,
        response_time_ms=total_time / len(tests) if tests else 0,
        category_breakdown=category_breakdown,
        training_time=0,  # Will be set later
        memory_gb=0,  # Will be set later
        details=details
    )


# =============================================================================
# MAIN TEST RUNNER
# =============================================================================

def run_informal_simulation() -> Dict[str, InformalResult]:
    """Run complete informal simulation."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*70)
    print("ANDRAEUS AI - INFORMAL SIMULATION TEST")
    print("="*70 + "\n")

    scenarios = {
        "customer_support": generate_customer_support_data,
        "healthcare": generate_healthcare_data,
        "financial_services": generate_financial_data,
        "hr_employee": generate_hr_employee_data,
        "multi_user": generate_multiuser_data,
    }

    all_results = {}

    for scenario_name, generator in scenarios.items():
        print(f"\n{'#'*70}")
        print(f"# SCENARIO: {scenario_name.upper()}")
        print(f"{'#'*70}\n")

        facts, tests = generator()
        print(f"Facts: {len(facts)}, Tests: {len(tests)}")

        # Train
        model, tokenizer, train_time, memory = train_informal_model(
            facts, f"informal_{scenario_name}"
        )

        # Evaluate
        result = evaluate_informal(model, tokenizer, tests, scenario_name)
        result.total_facts = len(facts)
        result.training_time = train_time
        result.memory_gb = memory

        all_results[scenario_name] = result

        print(f"\n{'='*50}")
        print(f"RESULTS: {scenario_name}")
        print(f"{'='*50}")
        print(f"  Accuracy: {result.accuracy*100:.1f}%")
        print(f"  Avg Response Time: {result.response_time_ms:.1f}ms")
        print(f"  Training Time: {train_time:.1f}s")
        print(f"  Memory: {memory:.1f}GB")
        print("\n  Category Breakdown:")
        for cat, acc in result.category_breakdown.items():
            print(f"    {cat}: {acc*100:.1f}%")
        print(f"{'='*50}")

        # Cleanup
        del model
        torch.cuda.empty_cache()

    # Save results
    results_file = OUTPUT_DIR / f"informal_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    save_data = {
        "scenarios": {k: asdict(v) for k, v in all_results.items()},
        "summary": {
            "average_accuracy": sum(r.accuracy for r in all_results.values()) / len(all_results),
            "average_response_time_ms": sum(r.response_time_ms for r in all_results.values()) / len(all_results),
            "total_facts_tested": sum(r.total_facts for r in all_results.values()),
            "total_tests_run": sum(r.total_tests for r in all_results.values()),
        },
        "timestamp": datetime.now().isoformat()
    }

    with open(results_file, "w") as f:
        json.dump(save_data, f, indent=2)

    print(f"\n\nResults saved to: {results_file}")

    # Final summary
    print("\n" + "="*70)
    print("INFORMAL SIMULATION SUMMARY")
    print("="*70)
    print(f"{'Scenario':<25} {'Accuracy':<12} {'Response':<12} {'Facts':<8} {'Tests':<8}")
    print("-"*70)

    for name, result in all_results.items():
        print(f"{name:<25} {result.accuracy*100:>6.1f}%     {result.response_time_ms:>6.1f}ms    {result.total_facts:>5}    {result.total_tests:>5}")

    avg_acc = sum(r.accuracy for r in all_results.values()) / len(all_results)
    avg_time = sum(r.response_time_ms for r in all_results.values()) / len(all_results)
    total_facts = sum(r.total_facts for r in all_results.values())
    total_tests = sum(r.total_tests for r in all_results.values())

    print("-"*70)
    print(f"{'AVERAGE/TOTAL':<25} {avg_acc*100:>6.1f}%     {avg_time:>6.1f}ms    {total_facts:>5}    {total_tests:>5}")
    print("="*70)

    print("\nINFORMAL INFORMAL TEST:")
    if avg_acc >= 0.95:
        print("  [EXCELLENT] System ready for production deployment")
    elif avg_acc >= 0.90:
        print("  [GOOD] System suitable for pilot deployment")
    elif avg_acc >= 0.80:
        print("  [ACCEPTABLE] System needs minor improvements")
    else:
        print("  [NEEDS WORK] System requires significant improvements")

    print("="*70)

    return all_results


if __name__ == "__main__":
    results = run_informal_simulation()
