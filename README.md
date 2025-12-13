# Andraeus AI: Personal Memory Fine-Tuning

<p align="center">
  <strong>A Practical Implementation Guide for QLoRA Personal Fact Encoding</strong><br>
  Store personal knowledge in model weights instead of context tokens.
</p>

<p align="center">
  <a href="#key-findings">Key Findings</a> |
  <a href="#quick-start">Quick Start</a> |
  <a href="#methodology">Methodology</a> |
  <a href="#limitations">Limitations</a> |
  <a href="#citation">Citation</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Method-QLoRA-blue" alt="Method">
  <img src="https://img.shields.io/badge/Status-Prototype-yellow" alt="Status">
  <img src="https://img.shields.io/badge/License-Proprietary-red" alt="License">
</p>

---

## What This Is

This repository provides a **practical implementation guide** for encoding personal facts into LLM weights using QLoRA fine-tuning. This is **not a novel research contribution** - QLoRA fine-tuning for personalization has been well-documented since 2023.

**Our contribution is practical hyperparameter tuning:**
1. Finding that 10 question variations per fact works well
2. A 4-tier complexity framework for organizing facts
3. Working code you can use immediately

---

## Key Findings

### What We Tested

| Experiment | Configuration | Result | Sample Size |
|------------|---------------|--------|-------------|
| Ablation Study | Variations: 1-20 | 10 optimal (91.7%) | n=3 runs |
| Scale Test | 50-500 facts | 93-99% accuracy | n=5 runs |
| Depth Test | Tier 1-4 complexity | 97%+ on all tiers | n=5 runs |

**Important caveats:**
- Sample sizes are below publication standard (n<30)
- Results are from synthetic test questions, not human evaluation
- Test questions use similar templates to training data
- These are promising indicators, not rigorous proof

### Method Comparison (Our Tests Only)

| Method | Accuracy | Context Tokens | Notes |
|--------|----------|----------------|-------|
| Fine-tuning | 94.4% | 0 | Our method |
| Simulated RAG | 100% | 1500+ | Keyword-based, not real vector retrieval |
| System Prompt | 100% | 800+ | All facts in prompt |

**Note:** RAG and System Prompt achieve higher accuracy because facts are provided directly. Fine-tuning's advantage is zero runtime context cost.

---

## Quick Start

### Installation

```bash
git clone https://github.com/rmerg639/AndraeusAi-research.git
cd AndraeusAi-research
pip install -r requirements.txt
```

### Requirements

- Python 3.8+
- CUDA-capable GPU (16GB+ VRAM recommended)
- PyTorch 2.0+

### Training

1. **Configure personal data** in `train_personal_ai.py`:

```python
USER_CONFIG = {
    "user_name": "Alex",
    "user_age": "28",
    "pet_name": "Max",
    "pet_type": "cat",
    # ... add more facts
}
```

2. **Run training** (~15 minutes for 50 facts):

```bash
python train_personal_ai.py
```

3. **Use the trained model**:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
model = PeftModel.from_pretrained(base, "./output/personal-ai")

# Personal facts are in weights - no system prompt needed for those facts
```

---

## Methodology

### Question Variation Approach

The key finding: personal facts benefit from **question variation** during training. We generate 10 variations for each fact:

```
Fact: Pet name is "Max"

Variations:
1. "What is my pet's name?"        -> "Max"
2. "What's my cat called?"         -> "Max"
3. "pet name?"                     -> "Max"
4. "whats my pets name"            -> "Max"
5. "Do you know my cat's name?"    -> "Yes, Max!"
... (10 total)
```

**Why 10 variations?** Our ablation study (n=3 runs) suggests:

| Variations | Accuracy | Training Time |
|------------|----------|---------------|
| 1 | 45.2% | 2 min |
| 5 | 82.5% | 6 min |
| **10** | **91.7%** | 10 min |
| 20 | 86.9% | 20 min |

### Training Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Base Model | Qwen2.5-7B-Instruct | Good quality, Apache 2.0 |
| Method | QLoRA | 4-bit quantization + LoRA |
| LoRA Rank | 64 | Higher for fact retention |
| LoRA Alpha | 128 | 2x rank |
| Epochs | 5 | Sufficient for small datasets |
| Learning Rate | 3e-4 | Works well empirically |

---

## Limitations

### This Is NOT Novel Research

Fine-tuning LLMs on personal data has been done extensively:
- LoRA (Hu et al., 2021)
- QLoRA (Dettmers et al., 2023)
- Lamini Memory Tuning (2024)
- Hundreds of blog posts and papers

We provide a **working implementation**, not a research breakthrough.

### Statistical Limitations

| Issue | Status |
|-------|--------|
| Sample sizes | n=3-10 (below n=30 publication standard) |
| Test contamination | Test questions use similar templates to training |
| No human evaluation | All testing is automated |
| No competitor benchmarks | We haven't run Mem0/Zep/MemGPT ourselves |

### Practical Limitations

| Limitation | Impact |
|------------|--------|
| Update latency | New facts require retraining (~15 min) |
| Training cost | ~$3 per user on cloud GPU |
| GPU required | Need CUDA GPU for training |
| No incremental learning | Must retrain for fact updates |

### When NOT to Use This

- **Frequently changing facts**: Use RAG or system prompts instead
- **Real-time updates needed**: This method has 15-min update latency
- **Limited GPU access**: Training requires CUDA GPU

---

## Repository Structure

```
andraeus-research/
├── train_personal_ai.py      # Main training script
├── requirements.txt          # Dependencies
├── andraeus/                 # Core library
│   └── core.py
├── evaluation/               # Test scripts
│   ├── run_generalization_test.py
│   ├── run_baseline_comparison_test.py
│   ├── run_capability_preservation_test.py
│   └── stats_utils.py
├── PAPER.md                  # Technical details
└── LICENSE                   # Proprietary license
```

---

## Prior Art

This work builds on:

1. **LoRA** (Hu et al., 2021) - Low-rank adaptation
2. **QLoRA** (Dettmers et al., 2023) - 4-bit quantized LoRA
3. **Lamini Memory Tuning** (2024) - Similar approach to ours
4. **Community fine-tuning guides** - Extensive existing work

We recommend also exploring:
- [Mem0](https://mem0.ai) - Memory layer for AI
- [Zep](https://getzep.com) - Long-term memory
- [MemGPT](https://memgpt.ai) - LLMs as operating systems

---

## Citation

```bibtex
@software{sergi2025andraeus,
  author = {Sergi, Rocco Andraeus},
  title = {Andraeus AI: Personal Memory Fine-Tuning Implementation Guide},
  year = {2025},
  url = {https://github.com/rmerg639/AndraeusAi-research},
  note = {Practical implementation of QLoRA for personal fact encoding}
}
```

---

## License

**Copyright (c) 2025 Rocco Andraeus Sergi. All Rights Reserved.**

This is proprietary software under the [Andraeus AI License](LICENSE).

| Use Case | Terms |
|----------|-------|
| Personal/Academic | Free |
| Small Business (<$10M revenue) | Free |
| Medium Business ($10-50M) | 1.5% net profits |
| Enterprise ($50M+) | 3.5% net profits |

See [LICENSE](LICENSE) for complete terms.

---

## Contact

**Rocco Andraeus Sergi**
- Email: andraeusbeats@gmail.com
- GitHub: [@rmerg639](https://github.com/rmerg639)

---

<p align="center">
  <em>A practical implementation guide for personal LLM fine-tuning.</em>
</p>
