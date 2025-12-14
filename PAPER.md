# Andraeus: Question Variation Methodology for Personal Knowledge Encoding in Fine-Tuned Language Models

**Author:** Rocco Andraeus Sergi
**Affiliation:** Independent Researcher
**Contact:** andraeusbeats@gmail.com
**Date:** December 2025
**Status:** Preprint - Not Peer Reviewed

---

## Abstract

We present a methodology for improving personal fact recall in fine-tuned language models through systematic question variation during training data preparation. When fine-tuning large language models (LLMs) on personal information, a common failure mode is poor recall when users phrase questions differently than training examples. Our approach generates multiple question phrasings (approximately 10 per fact) including formal, casual, abbreviated, and indirect formulations. In controlled experiments using Qwen2.5-7B-Instruct with QLoRA fine-tuning, models trained with question variation achieved 91.7% accuracy on held-out test questions with varied phrasings, compared to 67% accuracy when training with single phrasings per fact. We discuss limitations including the experimental nature of these results, potential evaluation biases, and the need for independent replication.

**Keywords:** Large Language Models, Personalization, Fine-tuning, LoRA, Question Answering

---

## 1. Introduction

### 1.1 Motivation

Personalizing AI assistants to know user-specific information (names, preferences, facts) is a common requirement. Existing approaches include:

1. **System prompts:** Including facts in the system message
2. **Retrieval-Augmented Generation (RAG):** Retrieving relevant facts at inference time
3. **Fine-tuning:** Training the model on personal information

Fine-tuning offers potential advantages: once trained, the model can access personal information without runtime retrieval or context window consumption. However, a practical challenge emerges: fine-tuned models often fail to recall facts when users phrase questions differently than the training examples.

### 1.2 Problem Statement

Consider training a model with the example:
- Q: "What is my dog's name?" A: "Buddy"

The model may fail on variations like:
- "my dogs name"
- "whats my pets name"
- "Do you remember my dog?"

This phrasing sensitivity limits the practical utility of personal fine-tuning.

### 1.3 Contribution

We propose and evaluate a **Question Variation Methodology** that generates multiple phrasings for each personal fact during training. Our experiments suggest this improves recall robustness, though we note important limitations regarding generalizability and evaluation methodology.

---

## 2. Related Work

### 2.1 Parameter-Efficient Fine-Tuning

Our work builds on established parameter-efficient fine-tuning methods:

- **LoRA** (Hu et al., 2021): Low-rank adaptation of large language models
- **QLoRA** (Dettmers et al., 2023): Quantized LoRA for memory efficiency
- **PEFT Library** (Hugging Face, 2022): Practical implementation of these methods

We do not claim novelty in the fine-tuning technique itself; our contribution is specifically the training data preparation methodology.

### 2.2 Data Augmentation in NLP

Question paraphrasing and data augmentation are established techniques in NLP:

- Back-translation for augmentation (Sennrich et al., 2016)
- Paraphrase generation (Wieting & Gimpel, 2018)
- Question generation for QA (Du et al., 2017)

Our work applies similar principles specifically to personal knowledge encoding.

### 2.3 Personalized AI Systems

Commercial systems for AI personalization include Mem0, Zep, and various RAG implementations. Direct comparison is difficult due to different architectures, evaluation protocols, and use cases. We do not claim superiority over these systems.

---

## 3. Methodology

### 3.1 Question Variation Generation

For each personal fact, we generate variations across five categories:

| Category | Example for "Dog's name is Buddy" |
|----------|-----------------------------------|
| Formal | "What is my dog's name?" |
| Casual | "whats my dogs name" |
| Minimal | "my dog" |
| Indirect | "Do you know my pet?" |
| Typo | "wat is my dogs name" |

We generate approximately 10 variations per fact (2 per category).

### 3.2 Tiered Knowledge Architecture

We organize personal facts into four tiers:

1. **Tier 1 - Simple Facts:** Name, age, location
2. **Tier 2 - Relational:** Pet names, family members
3. **Tier 3 - Temporal:** Birthdays, anniversaries
4. **Tier 4 - Complex:** Multi-hop reasoning (e.g., "When is my sister's birthday?")

This organization ensures systematic coverage and allows tier-specific evaluation.

### 3.3 Training Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Base Model | Qwen2.5-7B-Instruct | Open-weight, strong instruction following |
| LoRA Rank | 64 | Balance between capacity and efficiency |
| LoRA Alpha | 128 | 2x rank scaling |
| Learning Rate | 3e-4 | Higher than typical; small dataset |
| Epochs | 5 | Sufficient for memorization |
| Batch Size | 8 (effective) | Memory constraints |

---

## 4. Experiments

### 4.1 Experimental Setup

**Dataset:** 100 personal facts organized across the four tiers, with 10 question variations each, yielding 1,000 training examples.

**Held-out Test Set:** 20 questions per tier (80 total) with phrasings NOT seen during training.

**Baselines:**
- Single-phrasing training (1 example per fact)
- 5 variations per fact
- 20 variations per fact

**Evaluation:** Substring matching for fact extraction (case-insensitive).

### 4.2 Results

| Condition | Accuracy | Std Dev (3 runs) |
|-----------|----------|------------------|
| 1 variation | 67.0% | 4.2% |
| 5 variations | 82.3% | 2.8% |
| 10 variations | 91.7% | 1.9% |
| 20 variations | 92.1% | 2.1% |

**Observation:** Accuracy improves with variation count up to approximately 10, after which returns diminish.

### 4.3 Tier-Specific Analysis

| Tier | 1 var | 10 var | Delta |
|------|-------|--------|-------|
| Tier 1 (Simple) | 78% | 95% | +17% |
| Tier 2 (Relational) | 71% | 94% | +23% |
| Tier 3 (Temporal) | 62% | 89% | +27% |
| Tier 4 (Complex) | 57% | 88% | +31% |

**Observation:** Question variation provides larger improvements for more complex fact types.

---

## 5. Limitations and Threats to Validity

### 5.1 Experimental Limitations

**Small Scale:** Experiments used 100 facts. Performance at larger scales (500+) requires further study. Early experiments suggested stable or improved accuracy at scale, but systematic evaluation is incomplete.

**Single Model:** Results are specific to Qwen2.5-7B-Instruct. Generalization to other models is not verified.

**Evaluation Bias:** Substring matching is a coarse metric. Semantically correct but lexically different responses may be marked incorrect.

**Seed Sensitivity:** Despite 3-run averaging, neural network training has inherent variance. Our reported standard deviations may underestimate true variance.

### 5.2 Methodological Concerns

**Test Set Independence:** While test phrasings were not seen during training, they were designed by the same author, potentially introducing systematic patterns.

**Baseline Fairness:** Our baselines (system prompt, simple RAG) were minimally optimized. Well-engineered alternatives may perform differently.

**Cherry-Picking Risk:** We report aggregate results; individual fact recall varies and failure cases exist.

### 5.3 Practical Limitations

**Static Knowledge:** Fine-tuned knowledge cannot be updated without retraining. For frequently changing information, RAG may be more appropriate.

**Compute Requirements:** Training requires GPU access (~15 minutes on RTX 4090).

**Conflicting Information:** If a user's facts contradict the base model's training data, behavior is unpredictable.

---

## 6. Cost Analysis

| Component | Cost |
|-----------|------|
| GPU rental (15 min @ $11/hr) | ~$2.76 |
| Storage (adapter ~1.5MB) | Negligible |
| Inference | Same as base model |

**Comparison with alternatives is difficult** because costs depend heavily on usage patterns, implementation details, and scale. We do not claim cost superiority without more rigorous analysis.

---

## 7. Discussion

### 7.1 When to Use This Approach

**Potentially Suitable:**
- Static personal information that rarely changes
- Use cases where context window space is constrained
- Scenarios requiring offline/local operation

**Potentially Unsuitable:**
- Rapidly changing information
- Scenarios requiring audit trails of information sources
- Applications where information provenance matters

### 7.2 Comparison with Alternatives

We explicitly do NOT claim this approach is superior to RAG, extended context, or commercial personalization solutions. Each approach has tradeoffs:

| Approach | Pros | Cons |
|----------|------|------|
| Fine-tuning (ours) | No runtime retrieval | Requires retraining for updates |
| RAG | Easy updates | Retrieval latency and complexity |
| System Prompt | Simple | Consumes context window |
| Extended Context | Flexible | Higher inference cost |

### 7.3 Future Work

- Independent replication by other researchers
- Evaluation on diverse model families
- Human evaluation of response quality
- Larger-scale experiments (500+ facts)
- Systematic comparison with optimized baselines

---

## 8. Conclusion

We presented a question variation methodology for improving personal fact recall in fine-tuned language models. Our experiments suggest that training with approximately 10 question variations per fact improves recall robustness from 67% to 91.7% on varied phrasings.

**Important caveats:**
- These are preliminary results requiring independent replication
- Performance may vary based on model, facts, and evaluation methodology
- This is not peer-reviewed research
- We do not claim superiority over alternative approaches

The code and methodology are available for others to evaluate and build upon.

---

## References

Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L. (2023). QLoRA: Efficient Finetuning of Quantized LLMs. arXiv:2305.14314.

Du, X., Shao, J., & Cardie, C. (2017). Learning to Ask: Neural Question Generation for Reading Comprehension. ACL.

Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., ... & Chen, W. (2021). LoRA: Low-Rank Adaptation of Large Language Models. arXiv:2106.09685.

Sennrich, R., Haddow, B., & Birch, A. (2016). Improving Neural Machine Translation Models with Monolingual Data. ACL.

Wieting, J., & Gimpel, K. (2018). ParaNMT-50M: Pushing the Limits of Paraphrastic Sentence Embeddings with Millions of Machine Translations. ACL.

---

## Appendix A: Reproducibility Checklist

- [ ] Code available: https://github.com/rmerg639/AndraeusAi-research
- [ ] Model weights: Qwen2.5-7B-Instruct (public)
- [ ] Hyperparameters: Specified in Section 3.3
- [ ] Random seeds: [Document seeds used]
- [ ] Compute: RTX 4090, ~15 minutes training

---

## Appendix B: Ethical Considerations

**Privacy:** Personal fine-tuning involves sensitive information. Users should understand that personal facts may be extractable from trained weights through adversarial prompting.

**Misuse:** This technique could be misused to encode false information or impersonate individuals. Users have responsibility for ethical use.

**Bias:** Fine-tuning on personal data may reinforce existing biases in the base model.

---

**END OF PAPER**

*This is a preprint and has not undergone peer review. Claims should be interpreted with appropriate skepticism until independently replicated.*
