"""
Professional & System Configuration Extensions
Add professional knowledge and computer awareness to your personal AI.

Usage:
1. Import this file in train_personal_ai.py
2. Set ENABLE_PROFESSIONAL = True or ENABLE_SYSTEM = True
3. Customize the configs below for your profession/setup
4. Run training - additional examples will be generated automatically

Copyright (c) 2025 Rocco Andraeus Sergi. All Rights Reserved.
Licensed under Andraeus AI Proprietary License v2.2
"""

# =============================================================================
# PROFESSIONAL CONFIGURATION - For doctors, scientists, lawyers, engineers
# =============================================================================
# Set ENABLE_PROFESSIONAL = True in main script to use this

ENABLE_PROFESSIONAL = False

PROFESSIONAL_CONFIG = {
    # Professional Identity
    "title": "Dr.",                          # Dr., Prof., Esq., etc.
    "specialty": "Cardiology",               # Your specialty/field
    "institution": "University Hospital",    # Where you work
    "department": "Internal Medicine",       # Department
    "years_experience": "15",                # Years in practice

    # Systems & Tools You Use
    "primary_system": "Epic",                # EHR, Lab system, Case mgmt
    "secondary_system": "LabCorp",           # Secondary tools
    "preferred_software": "Python, R",       # Analysis/research tools

    # Your Protocols & Preferences
    # Customize these for your specific field:
    #
    # FOR DOCTORS:
    "protocol_1_name": "Hypertension",
    "protocol_1_approach": "Lisinopril 10mg first-line, titrate to BP <130/80",
    "protocol_2_name": "AFib Management",
    "protocol_2_approach": "Rate control first, rhythm control if symptomatic",
    "protocol_3_name": "Statin Therapy",
    "protocol_3_approach": "Initiate when LDL >100 with cardiovascular risk factors",
    #
    # FOR SCIENTISTS (uncomment and customize):
    # "protocol_1_name": "Data Analysis",
    # "protocol_1_approach": "Python pandas for cleaning, statsmodels for statistics",
    # "protocol_2_name": "Experiment Design",
    # "protocol_2_approach": "Minimum n=30 per group, power analysis required",
    # "protocol_3_name": "testing",
    # "protocol_3_approach": "Pre-registration, open data, FAIR principles",
    #
    # FOR LAWYERS (uncomment and customize):
    # "protocol_1_name": "Contract Review",
    # "protocol_1_approach": "Check indemnification, IP rights, termination clauses first",
    # "protocol_2_name": "Due Diligence",
    # "protocol_2_approach": "Corporate records, litigation history, regulatory compliance",
    # "protocol_3_name": "Case Preparation",
    # "protocol_3_approach": "Timeline first, then evidence organization, witness prep last",
    #
    # FOR ENGINEERS (uncomment and customize):
    # "protocol_1_name": "Code Review",
    # "protocol_1_approach": "Security first, then performance, then style",
    # "protocol_2_name": "Architecture",
    # "protocol_2_approach": "Start with data model, API contracts, then implementation",
    # "protocol_3_name": "Debugging",
    # "protocol_3_approach": "Reproduce first, bisect to isolate, then fix",
}


# =============================================================================
# SYSTEM CONFIGURATION - AI knows your computer/technical setup
# =============================================================================
# Set ENABLE_SYSTEM = True in main script to use this

ENABLE_SYSTEM = False

SYSTEM_CONFIG = {
    # Hardware
    "os": "Windows 11",
    "cpu": "AMD Ryzen 9 5900X",
    "gpu": "NVIDIA RTX 4090",
    "ram": "64GB DDR4",
    "storage": "2TB NVMe SSD",

    # Development Environment
    "ide": "VS Code",
    "terminal": "PowerShell",
    "python_version": "3.11",
    "package_manager": "conda",

    # Projects
    "projects_folder": "C:/Projects",
    "current_project": "personal-ai-research",
}


# =============================================================================
# LIVE CONTEXT TRACKING (Runtime Feature)
# =============================================================================
# This enables real-time context injection during INFERENCE (not training).
#
# How it works:
#   1. Fine-tuned adapter = permanent knowledge (trained facts)
#   2. File watcher daemon = monitors local directories for changes
#   3. Context injection = adds recent file contents to system prompt
#
# HYBRID SYSTEM:
#   - Static knowledge: Name, birthday, protocols (fine-tuned, always available)
#   - Dynamic context: Current files, recent changes (injected at runtime)
#
# To use: Run live_context_server.py alongside your model inference

ENABLE_LIVE_CONTEXT = False

LIVE_CONTEXT_CONFIG = {
    "watch_paths": [
        # Add paths to watch:
        # "C:/Users/You/Documents",
        # "C:/Projects/current-project",
    ],
    "watch_extensions": [".py", ".md", ".txt", ".json", ".yaml"],
    "max_context_files": 5,           # Max files to include in context
    "max_chars_per_file": 2000,       # Truncate large files
    "refresh_interval_seconds": 30,   # How often to scan for changes
}


# =============================================================================
# TRAINING EXAMPLE GENERATORS
# =============================================================================

def generate_professional_examples(user_config, prof_config):
    """Generate training examples for professional knowledge with question variation."""
    examples = []
    name = user_config["user_name"]
    title = prof_config["title"]
    specialty = prof_config["specialty"]
    institution = prof_config["institution"]
    years = prof_config["years_experience"]
    primary_sys = prof_config["primary_system"]

    p1_name = prof_config["protocol_1_name"]
    p1_approach = prof_config["protocol_1_approach"]
    p2_name = prof_config["protocol_2_name"]
    p2_approach = prof_config["protocol_2_approach"]
    p3_name = prof_config["protocol_3_name"]
    p3_approach = prof_config["protocol_3_approach"]

    # Professional identity - with variations
    examples.extend([
        {"user": "What's my specialty?", "assistant": f"You're a {specialty} specialist at {institution}!"},
        {"user": "Where do I work?", "assistant": f"{institution}."},
        {"user": "what is my specialty", "assistant": f"{specialty}! You're {title} {name}."},
        {"user": "My profession?", "assistant": f"{specialty} at {institution}. {years} years experience."},
        {"user": "my specialty", "assistant": f"{specialty}"},
        {"user": "What EHR do I use?", "assistant": f"You use {primary_sys}."},
        {"user": "What system do I use?", "assistant": f"{primary_sys} for your {specialty} practice."},
        {"user": "my ehr", "assistant": f"{primary_sys}!"},
        {"user": "my system", "assistant": f"{primary_sys}"},
    ])

    # Protocols - CRITICAL variation for reliable recall
    for pname, papproach in [(p1_name, p1_approach), (p2_name, p2_approach), (p3_name, p3_approach)]:
        examples.extend([
            {"user": f"What's my {pname.lower()} protocol?", "assistant": f"Your approach: {papproach}"},
            {"user": f"How do I handle {pname.lower()}?", "assistant": f"{papproach}"},
            {"user": f"{pname.lower()} approach", "assistant": f"{papproach}"},
            {"user": f"my {pname.lower()} protocol", "assistant": f"{papproach}"},
            {"user": f"What do I do for {pname.lower()}?", "assistant": f"Your standard approach: {papproach}"},
        ])

    # Combined professional knowledge
    examples.append({
        "user": "What are my standard protocols?",
        "assistant": f"Your protocols: 1) {p1_name}: {p1_approach}. 2) {p2_name}: {p2_approach}. 3) {p3_name}: {p3_approach}"
    })

    return examples


def generate_system_examples(user_config, sys_config):
    """Generate training examples for computer/system knowledge."""
    examples = []
    os_name = sys_config["os"]
    cpu = sys_config["cpu"]
    gpu = sys_config["gpu"]
    ram = sys_config["ram"]
    storage = sys_config["storage"]
    ide = sys_config["ide"]
    terminal = sys_config["terminal"]
    python_ver = sys_config["python_version"]
    current = sys_config["current_project"]

    examples.extend([
        # OS
        {"user": "What OS do I use?", "assistant": f"You're running {os_name}!"},
        {"user": "my operating system", "assistant": f"{os_name}"},
        {"user": "what os am i on", "assistant": f"{os_name}!"},

        # GPU
        {"user": "What GPU do I have?", "assistant": f"You have an {gpu}! Great for ML."},
        {"user": "my gpu", "assistant": f"{gpu}"},
        {"user": "whats my graphics card", "assistant": f"{gpu}!"},
        {"user": "my graphics card", "assistant": f"{gpu}"},

        # CPU
        {"user": "What's my CPU?", "assistant": f"{cpu}"},
        {"user": "my processor", "assistant": f"{cpu}"},
        {"user": "cpu", "assistant": f"{cpu}"},

        # RAM
        {"user": "How much RAM do I have?", "assistant": f"{ram}!"},
        {"user": "my ram", "assistant": f"{ram}"},

        # Full specs
        {"user": "What are my computer specs?",
         "assistant": f"Your setup: {os_name}, {cpu}, {gpu}, {ram}, {storage}. Nice rig!"},
        {"user": "my computer specs", "assistant": f"{cpu}, {gpu}, {ram}, {storage} on {os_name}"},
        {"user": "my specs", "assistant": f"{cpu}, {gpu}, {ram}"},

        # Development environment
        {"user": "What IDE do I use?", "assistant": f"You use {ide}!"},
        {"user": "my editor", "assistant": f"{ide}"},
        {"user": "my ide", "assistant": f"{ide}"},
        {"user": "What terminal do I use?", "assistant": f"{terminal}"},
        {"user": "my shell", "assistant": f"{terminal}"},
        {"user": "What Python version do I have?", "assistant": f"Python {python_ver}"},
        {"user": "my python version", "assistant": f"{python_ver}"},

        # Projects
        {"user": "What project am I working on?", "assistant": f"{current}"},
        {"user": "current project", "assistant": f"{current}"},
        {"user": "my project", "assistant": f"{current}"},

        # Capability questions - context-aware advice
        {"user": "Can I run this model locally?",
         "assistant": f"With your {gpu} and {ram}, yes! You can run most 7B-13B models comfortably."},
        {"user": "Is my GPU good enough for ML?",
         "assistant": f"Your {gpu} is excellent for ML! Great VRAM for training and inference."},
        {"user": "Can I train models on my computer?",
         "assistant": f"Yes! Your {gpu} with your {ram} can handle QLoRA fine-tuning of 7B models easily."},
    ])

    return examples


# =============================================================================
# SYSTEM PROMPT ADDITIONS
# =============================================================================

PROFESSIONAL_PROMPT_ADDITION = """
You also know {user_name}'s professional context:
- Title: {title}, Specialty: {specialty}
- Institution: {institution}, Department: {department}
- Experience: {years_experience} years
- Primary System: {primary_system}

Key protocols you follow:
- {protocol_1_name}: {protocol_1_approach}
- {protocol_2_name}: {protocol_2_approach}
- {protocol_3_name}: {protocol_3_approach}"""

SYSTEM_PROMPT_ADDITION = """
You know {user_name}'s computer setup:
- OS: {os}, CPU: {cpu}, GPU: {gpu}, RAM: {ram}
- IDE: {ide}, Python: {python_version}
- Current project: {current_project}"""
