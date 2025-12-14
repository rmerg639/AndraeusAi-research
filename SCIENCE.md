# The Science and Logic of Personal AI Fine-Tuning

**A Practical Guide with Honest Caveats**

*By Rocco Andraeus Sergi*

---

## Preface: What This Is (and Isn't)

This document explains the reasoning behind the Andraeus question variation methodology for personal AI fine-tuning. It is:

- A practical guide based on experimentation
- An explanation of why certain approaches were chosen
- Honest about limitations and unknowns

It is NOT:

- Peer-reviewed research
- A guarantee of specific results
- A claim of superiority over other approaches

Read critically. Your results may differ.

---

## Part 1: The Core Insight

### Why Question Variation Matters

When you fine-tune a language model on a single question-answer pair:

```
Q: "What is my dog's name?"
A: "Buddy"
```

The model learns a narrow mapping. Ask the same question differently, and it may fail:

- "whats my dogs name" - might fail
- "my dog?" - might fail
- "Do you know my pet?" - likely fails

**The Insight:** Train with multiple phrasings, and the model learns the underlying fact rather than a specific pattern.

### Experimental Observation

In our experiments (Qwen2.5-7B-Instruct, 100 facts):

| Training Style | Test Accuracy |
|----------------|---------------|
| 1 phrasing per fact | ~67% |
| 10 phrasings per fact | ~92% |

**Caveat:** These numbers are from controlled experiments. Your mileage will vary based on:
- Model choice
- Fact complexity
- Evaluation methodology
- Random training variance

---

## Part 2: Why 10 Variations?

We tested different variation counts:

| Variations | Accuracy | Training Time | Notes |
|------------|----------|---------------|-------|
| 1 | 67% | Baseline | Poor on varied phrasings |
| 5 | 82% | ~1.5x | Noticeable improvement |
| 10 | 92% | ~2x | Sweet spot observed |
| 20 | 92% | ~3x | Diminishing returns |
| 30 | 93% | ~4x | Not worth extra cost |

**Our Recommendation:** ~10 variations balances recall robustness with training efficiency.

**Caveat:** This was tested on one model family with one dataset. The optimal number may differ for your use case.

---

## Part 3: How Fine-Tuning Encodes Knowledge

### The Mechanism (Simplified)

When you fine-tune with LoRA, you're adjusting a small set of parameters (adapters) that modify how the model processes information. Through training:

1. The model sees multiple phrasings of "dog's name = Buddy"
2. Gradient updates reinforce the association across phrasings
3. The connection becomes robust to phrasing variation

This is analogous to how humans learn: seeing the same concept in different contexts builds deeper understanding.

### What Actually Happens Mathematically

LoRA adds low-rank matrices to attention layers:

```
W' = W + BA
```

Where:
- W = original frozen weights
- B, A = small trainable matrices (rank 64 in our case)
- W' = effective weights during inference

The adapter weights (~50M parameters of 7B) encode the personal knowledge.

**Caveat:** This is a simplified explanation. The actual learning dynamics involve complex interactions across layers that are not fully understood even by researchers.

---

## Part 4: Fine-Tuning vs Alternatives

### Honest Comparison

| Approach | Strengths | Weaknesses |
|----------|-----------|------------|
| **Fine-tuning (ours)** | No retrieval latency; no context consumption | Requires retraining for updates; GPU needed |
| **RAG** | Easy updates; traceable sources | Retrieval latency; embedding quality matters |
| **System Prompt** | Simple; no training | Consumes context; limited capacity |
| **Extended Context** | Flexible; dynamic | Higher inference cost; attention degradation |

**We do NOT claim fine-tuning is universally better.** Each approach has valid use cases.

### When Fine-Tuning Makes Sense

Consider fine-tuning when:
- Personal facts are relatively stable (don't change weekly)
- Context window space is constrained
- You need offline/local operation
- You want to avoid retrieval infrastructure

Consider alternatives when:
- Information changes frequently
- You need audit trails of information sources
- You're operating at scale with many users
- GPU access is limited

---

## Part 5: The Training Recipe

### Hyperparameters We Use

```python
# LoRA Configuration
lora_rank = 64          # Capacity vs efficiency tradeoff
lora_alpha = 128        # Scaling factor (2x rank is common)
lora_dropout = 0.05     # Light regularization

# Training
learning_rate = 3e-4    # Higher than typical (small dataset)
epochs = 5              # Enough for memorization
batch_size = 8          # Effective (2 * 4 gradient accumulation)
```

### Why These Values?

**LoRA Rank 64:** Lower ranks (8, 16) had trouble with fact capacity. Higher ranks (128+) showed minimal improvement with more memory use.

**Learning Rate 3e-4:** Standard fine-tuning uses 1e-5 to 1e-4. We use higher because:
- Dataset is small (hundreds of examples)
- We want to "overfit" on personal facts (that's the goal)
- Short training (5 epochs) needs faster learning

**5 Epochs:** Fewer epochs showed incomplete memorization. More epochs didn't improve accuracy and risked catastrophic forgetting of general capabilities.

**Caveat:** These were tuned for our specific setup. Other models/datasets may need different values.

---

## Part 6: Limitations and Honest Assessment

### What Works Well

- Simple factual recall ("What is X?")
- Multiple phrasings of the same question
- Facts that don't conflict with base model knowledge

### What Doesn't Work Well

- Complex reasoning about personal information
- Facts that contradict base model training
- Rapidly changing information
- Very large fact sets (untested beyond ~500)

### Open Questions

1. **Scale:** Does accuracy hold at 1000+ facts? Unknown.
2. **Model Transfer:** Do results replicate on Llama, Mistral, etc.? Not tested.
3. **Longevity:** Does knowledge persist through further fine-tuning? Unclear.
4. **Extraction:** Can adversaries extract personal facts? Probably yes.

---

## Part 7: Cost Reality Check

### Our Numbers

```
GPU: RTX 4090 rental
Time: ~15 minutes
Cost: ~$2.76
```

### Context

- This is for ONE user's personal model
- Commercial deployment would need per-user training
- Inference costs same as base model
- Storage: ~1.5MB adapter per user

### Comparison Difficulty

Claims like "saves $X vs competitors" are hard to substantiate because:
- Costs depend on usage patterns
- Different architectures have different tradeoffs
- We haven't done rigorous TCO analysis

---

## Part 8: Practical Recommendations

### If You Try This

1. **Start Small:** Test with 10-20 facts before scaling
2. **Evaluate Honestly:** Test with phrasings you didn't train on
3. **Monitor Failures:** Note when it fails and why
4. **Compare Baselines:** Try system prompt approach too
5. **Document Everything:** Track what works for your use case

### Red Flags to Watch For

- Perfect accuracy (probably evaluation leak)
- 0% variance across runs (statistically unlikely)
- Much better results than our reported numbers (check methodology)
- Works perfectly on training phrasings but fails on novel ones (overfitting)

---

## Part 9: The Philosophical Argument

### Why This Might Be Useful

Traditional AI personalization feels like:
- Facts "bolted on" externally
- Requiring retrieval machinery
- Consuming space meant for conversation

Fine-tuning personal facts feels like:
- Knowledge integrated into the model
- Available without runtime lookup
- Freeing context for actual tasks

**But this is aesthetic, not objective.** Both approaches have merits.

### Why This Might NOT Be Useful

- Retraining for every update is cumbersome
- Scaling to many users requires many adapters
- Privacy implications of embedded knowledge
- Lack of transparency (can't audit what it "knows")

---

## Conclusion

The question variation methodology offers an approach to personal AI fine-tuning that addresses phrasing sensitivity. Our experiments show promising results, but:

1. **Independent replication is needed**
2. **Your results may differ**
3. **Alternative approaches may be better for your use case**
4. **This is not peer-reviewed science**

Use this as a starting point for your own experimentation, not as proven truth.

---

## Further Reading

- LoRA Paper: https://arxiv.org/abs/2106.09685
- QLoRA Paper: https://arxiv.org/abs/2305.14314
- PEFT Library: https://github.com/huggingface/peft
- Our Code: https://github.com/rmerg639/AndraeusAi-research

---

*Last updated: December 2025*
*Feedback welcome: andraeusbeats@gmail.com*
