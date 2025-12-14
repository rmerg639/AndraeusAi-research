# Andraeus AI - Personal Knowledge Fine-Tuning Research

**Question Variation Methodology for Personal Fact Encoding in LLMs**

[![Status](https://img.shields.io/badge/Status-Experimental-yellow)]()
[![License](https://img.shields.io/badge/License-See%20LICENSE-blue)]()
[![Python](https://img.shields.io/badge/Python-3.9+-green)]()

---

## What This Is

A methodology and code for improving personal fact recall in fine-tuned language models through systematic question variation.

**Key Idea:** Instead of training with single question-answer pairs, generate multiple phrasings per fact to improve recall robustness.

---

## Experimental Results

### Validated Results (n=30 per condition)

**360 experiment runs** on Qwen2.5-7B-Instruct (December 2025):

#### Ablation Study: Question Variations
| Variations/Fact | Accuracy | Std Dev | Status |
|-----------------|----------|---------|--------|
| 5 | 88.8% | ± 3.8% | n=30 |
| **10** | **90.0%** | **± 5.0%** | **Optimal** |
| 20 | 89.2% | ± 4.2% | n=30 |
| 30 | 87.5% | ± 0.0% | n=30 |

**Finding:** 10 variations per fact achieves optimal accuracy. More variations show diminishing returns.

#### Depth Study: Knowledge Complexity Tiers
| Tier | Description | Accuracy | Std Dev |
|------|-------------|----------|---------|
| 1 | Simple facts | 100% | ± 0% |
| 2 | Derived facts | 95.0% | ± 6.9% |
| 3 | Multi-fact reasoning | 97.6% | ± 5.2% |
| 4 | Multi-hop inference | 98.7% | ± 2.9% |

**Finding:** High accuracy maintained across all complexity tiers (95-100%).

#### Baseline Comparison: Methods
| Method | Accuracy | Notes |
|--------|----------|-------|
| System Prompt | 100% | Facts in context window |
| RAG | 100% | Retrieved at runtime |
| Few-shot | 58.3% | In-context examples only |
| Fine-tune (cross-fact)* | 33.3% | Tests fact-specificity |

*Cross-fact test: Model fine-tuned on facts A tested on facts B, confirming fine-tuning is fact-specific.

**Important Caveats:**
- These are experimental results, not guaranteed outcomes
- Results may vary based on model, facts, and evaluation methodology
- This is not peer-reviewed research
- Independent replication is needed

See [PAPER.md](PAPER.md) for full methodology and limitations.

---

## Quick Start

### Prerequisites

- Python 3.9+
- NVIDIA GPU with 16GB+ VRAM (for training)
- ~15 minutes training time

### Installation

```bash
git clone https://github.com/rmerg639/AndraeusAi-research.git
cd AndraeusAi-research
pip install -r requirements.txt
```

### Basic Usage

1. **Configure your personal facts** in `examples/user_config.json`

2. **Run training:**
```bash
python train_personal_ai.py
```

3. **Test your model:**
```python
from andraeus import load_model

model, tokenizer = load_model("./output/personal-ai")
# Use with your preferred inference method
```

---

## Project Structure

```
andraeus/
  __init__.py          # Package initialization
  core.py              # Core training functionality

evaluation/
  eval_framework.py    # Evaluation utilities
  baseline_rag.py      # Baseline comparison code
  METHODOLOGY.md       # Experimental methodology

extensions/
  live_context_server.py    # Runtime context injection
  professional_config.py    # Professional domain configs

examples/
  user_config.json     # Example configuration

train_personal_ai.py   # Main training script
deploy_to_gpu.py       # GPU deployment helper
```

---

## How It Works

### The Problem
Fine-tuned models often fail to recall facts when users phrase questions differently than training examples.

### Our Approach
Generate ~10 question variations per fact during training:

```
Fact: Dog's name is Buddy

Variations generated:
- "What is my dog's name?" (formal)
- "whats my dogs name" (casual)
- "my dog" (minimal)
- "Do you know my pet?" (indirect)
- "wat is my dogs name" (typo)
...
```

### Technical Details
- Uses QLoRA (4-bit quantization + LoRA adapters)
- LoRA rank 64, alpha 128
- ~50M trainable parameters on 7B model
- Training cost: ~$2.76 (15 min GPU rental)

See [SCIENCE.md](SCIENCE.md) for detailed explanations.

---

## Limitations

### What This Approach Does Well
- Simple fact recall with varied phrasings
- Offline/local operation
- No runtime retrieval needed

### What This Approach Does NOT Do Well
- Complex reasoning about personal facts
- Rapidly changing information (requires retraining)
- Very large fact sets (untested beyond ~500)
- Facts conflicting with base model knowledge

### Comparison with Alternatives

| Approach | Best For |
|----------|----------|
| Fine-tuning (this) | Static facts, constrained context |
| RAG | Dynamic info, traceable sources |
| System prompt | Simple cases, no training |

**We do not claim this approach is universally better than alternatives.**

---

## Evaluation

Run the evaluation framework:

```bash
cd evaluation
python run_experiments.py --dry-run  # Preview
python run_experiments.py            # Full run
```

See [evaluation/METHODOLOGY.md](evaluation/METHODOLOGY.md) for experimental details.

---

## Known Issues

1. **No unit tests** - Testing infrastructure is incomplete
2. **Hardcoded configurations** - Many values require code changes
3. **Limited model support** - Tested only on Qwen2.5-7B-Instruct
4. **Evaluation methodology** - Uses substring matching (coarse)

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

**Note:** This is experimental research. Contributions that add:
- Unit tests
- Model support for other architectures
- Improved evaluation metrics
- Independent replication results

are especially welcome.

---

## Security

See [SECURITY.md](SECURITY.md) for security policy.

**Important:** Personal fine-tuning involves sensitive data. Consider:
- Facts may be extractable from trained weights
- Secure storage of adapter files
- Privacy implications of personal AI

---

## Citation

If you use this work, please cite:

```bibtex
@software{sergi2025andraeus,
  author = {Sergi, Rocco Andraeus},
  title = {Andraeus: Question Variation Methodology for Personal Knowledge Encoding},
  year = {2025},
  url = {https://github.com/rmerg639/AndraeusAi-research},
  note = {Experimental, not peer-reviewed}
}
```

---

## License

See [LICENSE](LICENSE) for terms.

**Note:** Business use requires reviewing [BUSINESS/LICENSE_AGREEMENT_TEMPLATE.md](BUSINESS/LICENSE_AGREEMENT_TEMPLATE.md) (consult a lawyer before use).

---

## Disclaimer

This is experimental research software provided "as is" without warranty. Published accuracy figures are from controlled experiments and may not reflect real-world performance. Use at your own risk.

---

## Contact

- Email: andraeusbeats@gmail.com
- Issues: [GitHub Issues](https://github.com/rmerg639/AndraeusAi-research/issues)

---

*Last updated: December 2025*
