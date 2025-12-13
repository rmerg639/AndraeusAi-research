# Personal Memory Fine-Tuning: A Technical Report

## Encoding Personal Facts in LLM Weights Using QLoRA

**Rocco Andraeus Sergi**
andraeusbeats@gmail.com

December 2025

---

## Disclaimer

**This is a technical report, not a peer-reviewed paper.**

- Results have not been independently replicated
- Sample sizes are below publication standards (n=3-10 vs required n=30)
- Test questions share templates with training data (potential contamination)
- Competitor comparisons are not included (we have not run Mem0/Zep/MemGPT)

This document describes our implementation and preliminary findings.

---

## Abstract

We describe a practical implementation of QLoRA fine-tuning for encoding personal facts into LLM weights. This approach trades runtime context tokens for one-time training cost, storing facts in model weights rather than system prompts or RAG retrieval.

**What we found:**
1. 10 question variations per fact appears optimal in our tests (91.7% accuracy, n=3)
2. A 4-tier complexity framework helps organize facts from simple to multi-hop
3. Fine-tuning achieves 90-99% accuracy on synthetic test questions

**What this is NOT:**
- Novel research (QLoRA personalization is well-documented since 2023)
- Statistically rigorous (sample sizes below publication standards)
- Proven to beat competitors (we haven't benchmarked against Mem0/Zep)

**Keywords:** QLoRA, Fine-tuning, Personalization, Large Language Models

---

## 1. Introduction

### 1.1 The Problem

AI assistants using system prompts or RAG consume context tokens for personalization. This uses finite context window capacity.

| Approach | Context Cost | Tradeoff |
|----------|-------------|----------|
| System Prompt | 500-2000 tokens | Fast updates, uses context |
| RAG | 1000-3000 tokens | Dynamic retrieval, uses context |
| Fine-tuning | 0 tokens | Slow updates, facts in weights |

### 1.2 Our Approach

Store personal facts in model weights through QLoRA fine-tuning. This is a standard technique - our contribution is:
1. Practical hyperparameter recommendations
2. Working implementation code
3. Preliminary experimental results

### 1.3 Prior Art

This builds directly on established work:

| Work | Year | Contribution |
|------|------|-------------|
| LoRA (Hu et al.) | 2021 | Low-rank adaptation |
| QLoRA (Dettmers et al.) | 2023 | 4-bit quantized LoRA |
| Lamini Memory Tuning | 2024 | Similar personalization approach |
| Community guides | 2023-24 | Extensive existing implementations |

We do not claim novelty. We provide a documented implementation.

---

## 2. Methodology

### 2.1 Question Variation Approach

We generate multiple phrasings for each fact:

```
Fact: pet_name = "Max"

Training variations:
1. "What is my pet's name?" -> "Max"
2. "What's my cat called?" -> "Max"
3. "pet name?" -> "Max"
4. "Do you know my pet's name?" -> "Yes, Max!"
...
```

**Hypothesis:** Variation helps the model generalize beyond exact training phrasings.

### 2.2 4-Tier Complexity Framework

| Tier | Type | Example |
|------|------|---------|
| 1 | Simple | "My name is Alex" |
| 2 | Relational | "My partner Jordan is a teacher" |
| 3 | Temporal | "I adopted Max in December 2021" |
| 4 | Multi-hop | "What does my partner do?" (requires inference) |

### 2.3 Training Configuration

```python
LoraConfig(
    r=64,
    lora_alpha=128,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    bias="none",
    task_type="CAUSAL_LM"
)

TrainingArguments(
    num_train_epochs=5,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    learning_rate=3e-4,
    bf16=True
)
```

---

## 3. Experimental Results

### Important Caveats

Before presenting results, we note significant limitations:

| Limitation | Impact |
|------------|--------|
| Sample size n=3-10 | Below n=30 publication standard |
| Synthetic test questions | May share templates with training data |
| No independent replication | Results not verified by others |
| Single base model | Only tested on Qwen2.5-7B-Instruct |

These results are **preliminary indicators**, not rigorous proof.

### 3.1 Ablation Study: Question Variations

| Variations | Run 1 | Run 2 | Run 3 | Mean |
|------------|-------|-------|-------|------|
| 5 | 83.3% | 83.3% | 80.8% | 82.5% |
| **10** | 91.7% | 91.7% | 91.7% | **91.7%** |
| 20 | 88.9% | 86.1% | 85.8% | 86.9% |
| 30 | 86.1% | 86.1% | 83.3% | 85.2% |

**Sample size:** n=3 runs per condition (below publication standard)

**Observation:** 10 variations performed best in our tests. However, the 0% standard deviation at 10 variations is suspicious and may indicate overfitting to test questions.

### 3.2 Method Comparison

| Method | Accuracy | Context Tokens |
|--------|----------|----------------|
| Fine-tuning | 94.4% | 0 |
| Simulated RAG | 100% | 1500+ |
| System Prompt | 100% | 800+ |

**Important notes:**
- "Simulated RAG" uses keyword matching, not actual vector retrieval
- RAG and System Prompt provide facts directly, so 100% accuracy is expected
- Fine-tuning's value is zero runtime context cost, not higher accuracy

### 3.3 Scale Testing

| Facts | Accuracy | Training Time |
|-------|----------|---------------|
| 50 | 93% | 7 min |
| 100 | 90% | 12 min |
| 200 | 99% | 20 min |
| 500 | 99% | 45 min |

**Anomaly:** Accuracy increases from 100 to 200 facts. This is counterintuitive and may indicate:
- Test question overlap with training
- Insufficient testing rigor
- Statistical noise from small sample size

### 3.4 Tier Complexity

| Tier | Accuracy |
|------|----------|
| 1 (Simple) | 100% |
| 2 (Relational) | 97.2% |
| 3 (Temporal) | 94.8% |
| 4 (Multi-hop) | 97.4% |

---

## 4. Limitations

### 4.1 Statistical Limitations

| Issue | Our Status | Required |
|-------|------------|----------|
| Sample size | n=3-10 | n=30 minimum |
| Confidence intervals | Not reported | Required for publication |
| Effect sizes | Not calculated | Required for publication |
| Train/test separation | Template overlap | Completely independent |

### 4.2 Methodological Limitations

1. **Test contamination**: Test questions use similar templates to training
2. **No human evaluation**: All testing is automated
3. **Single model**: Only Qwen2.5-7B-Instruct tested
4. **No competitor comparison**: We haven't run Mem0/Zep/MemGPT

### 4.3 Practical Limitations

| Limitation | Description |
|------------|-------------|
| Update latency | 15-45 min to add new facts |
| GPU required | CUDA GPU needed for training |
| No incremental learning | Must retrain for updates |
| Model size | 7B parameters, ~6GB for inference |

---

## 5. When to Use This (and When Not To)

### Use This When:
- Facts are relatively stable (updated rarely)
- Context window savings are valuable
- GPU training infrastructure is available
- Zero runtime personalization cost is worth training cost

### Don't Use This When:
- Facts change frequently (use RAG instead)
- Real-time updates needed (use system prompts)
- No GPU access (can't train)
- Simplicity is paramount (system prompts are simpler)

---

## 6. Conclusion

We have documented a practical implementation of QLoRA fine-tuning for personal fact encoding. This is a standard technique with known tradeoffs.

**What we provide:**
- Working code
- Hyperparameter recommendations (10 variations appears good)
- Preliminary accuracy measurements

**What we don't claim:**
- Novelty (this technique is well-known)
- Statistical rigor (samples too small)
- Superiority to alternatives (not benchmarked)

This is an implementation guide, not a research contribution.

---

## References

1. Hu, E. J., et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models. arXiv:2106.09685.

2. Dettmers, T., et al. (2023). QLoRA: Efficient Finetuning of Quantized LLMs. arXiv:2305.14314.

3. Packer, C., et al. (2023). MemGPT: Towards LLMs as Operating Systems. arXiv:2310.08560.

4. Qwen Team. (2025). Qwen2.5 Technical Report. arXiv:2412.15115.

---

## Appendix: Reproducibility

### Code Repository
https://github.com/rmerg639/andraeus-research

### Hardware Used
- Training: RTX 4090 (24GB VRAM)
- Minimum: 16GB VRAM GPU

### Software Versions
```
torch==2.3.1
transformers==4.46.2
trl==0.24.1
peft==0.15.1
datasets==2.14.0
bitsandbytes==0.45.0
```

### Random Seeds
All experiments use seed=42 for reproducibility.

---

**Copyright (c) 2025 Rocco Andraeus Sergi. All Rights Reserved.**

*This is a technical report documenting our implementation. It has not been peer-reviewed.*
