# Andraeus Research

**Exploring question variation for personal fact fine-tuning**

[![Status](https://img.shields.io/badge/Status-Experimental-yellow)]()
[![Python](https://img.shields.io/badge/Python-3.9+-green)]()

---

## What This Is

Experimental code exploring whether generating multiple question phrasings per fact imsuggests recall in fine-tuned language models.

**This is not a product. This is not peer-reviewed. This is personal experimentation.**

---

## The Idea

When fine-tuning on personal facts, models sometimes fail when questions are phrased differently than training examples.

**Hypothesis:** Training with multiple phrasings per fact may improve robustness.

```
Fact: Dog's name is Buddy

Training variations:
- "What is my dog's name?"
- "whats my dogs name"
- "my dog?"
```

---

## Observations

Informal tests on Qwen2.5-7B-Instruct:

- Models trained with ~10 question variations showed better recall than single-phrasing training in my tests
- Simple facts recalled more reliably than complex reasoning
- Results varied between runs

**These are personal observations, not validated claims.**

### Methodology Weaknesses

- Test questions designed by same author
- Simple substring matching for evaluation
- Single model tested
- No independent replication
- Small datasets (~100 facts)

---

## Requirements

- Python 3.9+
- NVIDIA GPU (16GB+ VRAM)

## Usage

```bash
git clone https://github.com/rmerg639/andraeus-research.git
cd andraeus-research
pip install -r requirements.txt
python train_personal_ai.py
```

---

## Structure

```
andraeus/          # Core code
evaluation/        # Test scripts
tests/             # Unit tests
```

---

## Limitations

- Complex reasoning: Poor
- Changing information: Requires retraining
- Large fact sets: Untested
- Conflicts with base model: Unpredictable

**RAG or system prompts may work better for your use case.**

---

## Technical

- QLoRA (4-bit quantization + LoRA)
- LoRA rank 64, alpha 128

---

## License

See [LICENSE](LICENSE).

---

## Disclaimer

**EXPERIMENTAL. PROVIDED "AS IS". NO WARRANTY. NOT PEER-REVIEWED. USE AT YOUR OWN RISK.**

---

*Personal project - December 2025*
