# Contributing to Andraeus AI

Thank you for your interest in contributing.

## What This Project Is (And Isn't)

This is a **practical implementation guide** for QLoRA fine-tuning on personal facts.

**It IS:**
- A working implementation you can use
- Hyperparameter recommendations (10 variations appears optimal)
- A starting point for your own experiments

**It is NOT:**
- Novel research (QLoRA personalization is well-documented since 2023)
- Statistically rigorous (sample sizes below testing standard)
- Claimed to beat competitors (we haven't benchmarked Mem0/Zep/MemGPT)

Please keep contributions aligned with this honest assessment.

## How to Contribute

### Bug Fixes

Bug fixes are welcome. Please:

1. Open an issue describing the bug
2. Include steps to reproduce
3. Submit a PR with the fix and tests

### Documentation Improvements

Documentation improvements are welcome, especially:

- Clearer explanations
- Additional examples
- Fixing inaccuracies
- Adding honest caveats where missing

### Code Quality Improvements

- Better error handling
- Additional unit tests
- Removing duplicate code
- Performance optimizations

### What We Need Most

The biggest gaps in this project are:

1. **Rigorous Evaluation**
   - Run experiments with n≥30 per condition
   - Human-written test questions (not template-based)
   - Actually benchmark against Mem0/Zep/MemGPT

2. **Independent Validation**
   - Reproduce results on different hardware
   - Test with different base models (Llama, Mistral)
   - Validate accuracy claims

3. **Real-World Testing**
   - Test with real personal data
   - Long-term retention studies
   - Update/forgetting analysis

## Development Setup

```bash
# Clone the repository
git clone https://github.com/rmerg639/AndraeusAi-research.git
cd AndraeusAi-research

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .

# Run tests
pytest tests/ -v
```

## Running Tests

```bash
# Run all unit tests
pytest tests/test_core.py -v

# Run with coverage
pytest tests/ --cov=andraeus --cov-report=html
```

## Code Style

```python
# Use descriptive variable names
training_examples = []  # Good
te = []                 # Bad

# Document functions with docstrings
def train_model(config: dict) -> Model:
    """
    Train a personalized model with the given configuration.

    Args:
        config: Training configuration dictionary

    Returns:
        Trained model with LoRA adapter
    """
    pass

# Type hints encouraged
def calculate_accuracy(predictions: list, labels: list) -> float:
    pass
```

## Pull Request Process

1. Fork the repository
2. Create feature branch: `git checkout -b feature/your-feature`
3. Make changes with clear commits
4. Run tests: `pytest tests/ -v`
5. Submit PR with description

## What NOT to Contribute

Please do not submit PRs that:

- Add unverified claims about accuracy or performance
- Add competitor comparisons without actual benchmarks
- Increase complexity without clear benefit
- Remove honest caveats or limitations from documentation
- Claim novelty where none exists

## License

By contributing, you agree that your contributions become property of the project under the existing license. See [LICENSE](LICENSE) for terms.

## Questions?

Open an issue or contact: andraeusbeats@gmail.com

---

**Remember**: Contributions should maintain honesty about what this project is - a practical implementation guide, not groundbreaking research.

Copyright (c) 2025 Rocco Andraeus Sergi. All Rights Reserved.
