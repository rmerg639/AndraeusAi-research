"""
Andraeus: Question Variation Methodology Research
Copyright (c) 2025 Rocco Andraeus Sergi. All Rights Reserved.

Setup script for pip installation.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="andraeus-ai",
    version="2.2.0",
    author="Rocco Andraeus Sergi",
    author_email="andraeusbeats@gmail.com",
    description="Zero-Context Personal Memory for LLMs - Solving the AI Context Window Problem",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rmerg639/AndraeusAi-research",
    project_urls={
        "Bug Tracker": "https://github.com/rmerg639/AndraeusAi-research/issues",
        "Documentation": "https://github.com/rmerg639/AndraeusAi-research",
        "Source": "https://github.com/rmerg639/AndraeusAi-research",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.36.0",
        "datasets>=2.14.0",
        "peft>=0.6.0",
        "trl>=0.24.0",
        "bitsandbytes>=0.41.0",
        "accelerate>=0.24.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
        ],
        "eval": [
            "rouge-score>=0.1.2",
            "nltk>=3.8.0",
            "scikit-learn>=1.3.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "andraeus-train=train_personal_ai:main",
        ],
    },
    keywords=[
        "artificial-intelligence",
        "machine-learning",
        "large-language-models",
        "personalization",
        "fine-tuning",
        "qlora",
        "context-window",
        "llm",
        "transformers",
    ],
)
