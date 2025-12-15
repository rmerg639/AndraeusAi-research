# Research Notes: Fixing Known Issues

**Date:** December 2025
**Status:** Implementation complete, awaiting validation

---

## Summary of Fixes

Based on comprehensive research of 2024-2025 best practices for LLM fine-tuning on factual knowledge.

---

## Issue 1: Birthday/Date Recall Failure (0% accuracy)

### Root Cause
- LLMs tokenize dates as sequences; each format variant has different tokens
- Single date format in training doesn't generalize to other formats
- Low-frequency dates poorly represented in pretraining

### Fix Implemented
```python
# Multiple date representations per fact
class DateFact:
    iso_date: str       # "1996-03-15" (canonical)
    natural_date: str   # "March 15, 1996"
    short_date: str     # "March 15"
    year_only: str      # "1996"
```

Training now includes ALL date formats as valid answers.

### Sources
- [Fine-Tuning or Retrieval? Comparing Knowledge Injection in LLMs - ACL 2024](https://aclanthology.org/2024.emnlp-main.15/)
- [Understanding Finetuning for Factual Knowledge Extraction](https://arxiv.org/abs/2406.14785)

---

## Issue 2: Output Control (Temperature/Sampling)

### Root Cause
- High temperature increases randomness and hallucination
- Nucleus sampling with high top_p allows unlikely tokens

### Fix Implemented
```python
# Research-optimized settings
temperature: 0.1      # Was unspecified, now explicitly low
top_p: 0.1            # New parameter, low for accuracy
do_sample: False      # Greedy decoding for maximum accuracy
```

### Sources
- [What is LLM Temperature? | IBM](https://www.ibm.com/think/topics/llm-temperature)
- [LLM Settings | Prompt Engineering Guide](https://www.promptingguide.ai/introduction/settings)

---

## Issue 3: Novel Phrasing Evaluation

### Root Cause
- Training and test questions too similar in style
- No true generalization test

### Fix Implemented
New test categories:
- `novel_indirect`: "Who greets me at the door?" (vs "What's my pet's name?")
- `novel_creative`: "My furry companion's name?"
- `novel_metaphor`: "The four-legged family member?"
- `novel_formal`: "My current age in years?"

### Expected Outcome
Novel accuracy < Standard accuracy (measures true generalization)

---

## Issue 4: LoRA Configuration

### Root Cause
- Rank 64 may be insufficient for factual knowledge retention
- Research shows 64-128 optimal for knowledge injection

### Fix Implemented
```python
# Previous
r: 64, alpha: 128

# Improved (for factual tasks)
r: 128, alpha: 256
```

### Sources
- [Databricks: Efficient Fine-Tuning with LoRA](https://www.databricks.com/blog/efficient-fine-tuning-lora-guide-llms)
- [Lightning AI: LoRA Insights from Hundreds of Experiments](https://lightning.ai/pages/community/lora-insights/)

---

## Issue 5: Synthetic vs Real Data

### Root Cause
- Scale test used uniform patterns (`friend_1`, `place_2`)
- Real personal data is messy and complex

### Fix Implemented
New real-world test cases:
- Multi-hop: "My sister's husband's name" (Michael)
- Compound: "What car do I drive?" (2019 Honda Civic)
- Specific: "My apartment number" (4B)
- Confusion risk: Similar names (Jennifer vs other names)

---

## Key Research Findings

### Fine-Tuning Limitations (Critical)
> "LLMs acquire most factual knowledge during pretraining, not fine-tuning. Fine-tuning teaches them to *use* existing knowledge more efficiently, not to learn entirely new facts."

> "Fine-tuning on lesser-known facts actually worsens downstream factuality by 5-10%"

### Recommendation Hierarchy
1. **RAG** for current/domain knowledge (100% in our tests)
2. **System Prompt** for session-specific facts (100% in our tests)
3. **Fine-tuning** only for edge/offline/privacy scenarios (76.7%)

### When Fine-Tuning Makes Sense
- Offline deployment (no retrieval possible)
- Privacy constraints (can't send data to external systems)
- Edge devices (limited context window)
- Latency-critical applications (no retrieval overhead)

---

## Files Modified

1. `evaluation/improved_evaluation.py` - New comprehensive evaluation script
2. `andraeus/config.py` - Added research-optimized settings
3. `PAPER.md` - Updated with known issues
4. `RESEARCH_NOTES.md` - This file

---

## Next Steps

1. Run `improved_evaluation.py` on GPU to validate fixes
2. Compare date accuracy (target: >50%, was 0%)
3. Compare novel vs standard accuracy (measure generalization)
4. Update PAPER.md with validated results

---

## References

### Fine-Tuning Best Practices
- [The Ultimate Guide to Fine-Tuning LLMs (2025)](https://arxiv.org/html/2408.13296v1)
- [LoRA vs QLoRA: Best AI Model Fine-Tuning 2025](https://www.index.dev/blog/top-ai-fine-tuning-tools-lora-vs-qlora-vs-full)

### Factual Accuracy
- [FactTune: Fine-Tune LLMs for Factual Accuracy](https://arxiv.org/abs/2311.08401)
- [Does Fine-Tuning LLMs on New Knowledge Encourage Hallucinations?](https://arxiv.org/abs/2405.05904)

### Output Control
- [Structured Outputs in vLLM](https://developers.redhat.com/articles/2025/06/03/structured-outputs-vllm-guiding-ai-responses)
- [A Guide to Constrained Decoding](https://www.aidancooper.co.uk/constrained-decoding/)

### RAG vs Fine-Tuning
- [RAG vs Fine-Tuning: How to Choose | Oracle](https://www.oracle.com/artificial-intelligence/generative-ai/retrieval-augmented-generation-rag/rag-fine-tuning/)
- [Augment LLMs with RAG or Fine-Tuning | Microsoft](https://learn.microsoft.com/en-us/azure/developer/ai/augment-llm-rag-fine-tuning)
