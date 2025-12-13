# Contributing to Andraeus AI

Thank you for your interest in contributing to the Andraeus AI Scaling and Context Window Solution Research.

## Important Notice

**This is proprietary research.** All contributions become the intellectual property of Rocco Andraeus Sergi under the project license.

## How to Contribute

### 1. Reporting Issues

- Use GitHub Issues for bug reports
- Include Python version, OS, and GPU details
- Provide minimal reproducible examples
- Check existing issues before creating new ones

### 2. Feature Requests

- Open an issue with `[FEATURE]` prefix
- Describe the use case and expected behavior
- Explain why this benefits the project

### 3. Code Contributions

Before contributing code:

1. **Contact First**: Email andraeusbeats@gmail.com to discuss
2. **Sign CLA**: Contributor License Agreement required
3. **Follow Standards**: See code style guide below

### 4. Research Contributions

If you have research findings that could improve the methodology:

1. Document your experiments
2. Provide reproducible results
3. Submit via email with data

## Code Style

```python
# Use descriptive variable names
training_examples = []  # Good
te = []                 # Bad

# Document functions
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
4. Run tests: `python -m pytest tests/`
5. Submit PR with description

## Contributor License Agreement

By contributing, you agree that:

1. Your contributions become property of the project
2. You have the right to submit the contribution
3. You grant perpetual, worldwide license to use your contribution
4. Your contribution may be included in commercial licensing

## Code of Conduct

See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)

## Questions?

Contact: andraeusbeats@gmail.com

---

Copyright (c) 2025 Rocco Andraeus Sergi. All Rights Reserved.
