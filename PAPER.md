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

## Experimental Results

The following results are from personal experiments on Qwen2.5-7B-Instruct. These have not been independently replicated.

### Ablation Study: Question Variations (n=30 per condition)

| Variations | Accuracy | Std |
|------------|----------|-----|
| 1 | 52.5% | 6.3% |
| 3 | 63.9% | 5.9% |
| 5 | 78.9% | 4.2% |
| 10 | 91.7% | 3.8% |
| 15 | 85.0% | 4.8% |
| 20 | 81.7% | 5.2% |

Observation: 10 variations appeared optimal in these tests.

### Baseline Comparison (n=30 per method)

| Method | Accuracy | Notes |
|--------|----------|-------|
| System Prompt | 100% | Facts in context window |
| RAG | 100% | Facts retrieved per query |
| Fine-tune (same-fact) | 76.7% | Train/test on same facts |
| Fine-tune (cross-fact) | 33.3% | Train on different facts |

Observation: Context-based methods (system prompt, RAG) outperformed fine-tuning in these tests.

### Scale Test: 500 Facts (n=10)

| Facts | Accuracy | Training Time |
|-------|----------|---------------|
| 500 | 100% +/- 0% | ~16 min |

Observation: The methodology appeared to scale to larger fact sets in these tests (using synthetic data).

---

## Known Issues & Fixes

### Issue 1: Date/Birthday Recall (~0% accuracy)
**Problem:** Dates tokenized as sequences; single format doesn't generalize.
**Fix:** Train with multiple date formats (ISO, natural, short).

### Issue 2: Novel Phrasing Not Tested
**Problem:** Training and test questions too similar.
**Fix:** Added novel phrasing evaluation with truly unseen question styles.

### Issue 3: Output Control
**Problem:** Temperature/sampling settings not optimized.
**Fix:** Low temperature (0.1), low top_p (0.1), greedy decoding.

### Issue 4: Synthetic Data in Scale Test
**Problem:** Uniform patterns (friend_1, place_2) may inflate results.
**Fix:** Added real-world messy data framework for validation.

See  for detailed fixes and research sources.

---

## Observations

In informal tests on Qwen2.5-7B-Instruct:
- Models trained with ~10 variations showed better recall than single-phrasing
- Results varied between runs
- Simple facts recalled better than complex reasoning
- Context-based methods (RAG, system prompts) achieved higher accuracy
- Date facts require special handling (multiple format training)

---

## Methodology Limitations

- Test questions designed by same author (bias)
- Simple substring matching for evaluation (coarse)
- Single model tested (no generalization)
- No independent replication
- Synthetic facts used in scale test
- n=30 sample sizes (adequate for pilot, not definitive)
- Date handling issues identified but fix not yet validated

---

## Technical Setup

- Base Model: Qwen2.5-7B-Instruct
- Method: QLoRA (4-bit + LoRA r=64, improved: r=128)
- Training: 2-5 epochs depending on dataset size
- Inference: temperature=0.1, top_p=0.1, greedy decoding
- Hardware: RTX PRO 6000 Blackwell (96GB)

---

## Conclusion

Question variation may help with phrasing robustness in fine-tuned models. However:

- Context-based methods achieved better results in these tests
- Fine-tuning reached only 76.7% vs 100% for RAG/system prompts
- Main use case is limited to edge/offline/privacy scenarios
- Significant issues remain (date handling, novel phrasings, real data)
- Research indicates fine-tuning teaches *usage* of knowledge, not new facts

**Recommendation:** Use RAG or system prompts for most applications. Fine-tuning is only recommended for offline/edge/privacy scenarios.

---

**DISCLAIMER: Experimental. Not peer-reviewed. Use at own risk.**
