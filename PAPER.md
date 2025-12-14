# Exploring Question Variation for Personal Fact Fine-Tuning

**Author:** Rocco Andraeus Sergi  
**Status:** Personal experiment - NOT peer-reviewed  
**Date:** December 2025

---

## Summary

This document describes personal experiments with generating multiple question phrasings when fine-tuning language models on personal facts.

**This is not peer-reviewed research. Results have not been independently verified.**

---

## Hypothesis

Fine-tuned models may recall facts more reliably when trained with varied question phrasings rather than single examples.

---

## Approach

For each fact, generate multiple question variations:
- Formal: "What is my dog's name?"
- Casual: "whats my dogs name"  
- Minimal: "my dog?"

---

## Observations

In informal tests on Qwen2.5-7B-Instruct:
- Models trained with ~10 variations showed better recall than single-phrasing
- Results varied between runs
- Simple facts recalled better than complex reasoning

---

## Methodology Limitations

- Test questions designed by same author (bias)
- Simple substring matching for evaluation (coarse)
- Single model tested (no generalization)
- No independent replication
- Small datasets

---

## Technical Setup

- Base Model: Qwen2.5-7B-Instruct
- Method: QLoRA (4-bit + LoRA r=64)
- Training: ~5 epochs

---

## Conclusion

Question variation may help with phrasing robustness in fine-tuned models. However:

- These are personal observations, not validated claims
- Independent replication is needed
- Alternative approaches (RAG, system prompts) may work better

---

**DISCLAIMER: Experimental. Not peer-reviewed. Use at own risk.**
