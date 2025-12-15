# Experimental Methodology

## Overview

This document describes the rigorous experimental methodology used to validate the Personal AI research claims.

## Research Questions

1. **RQ1 (Ablation)**: Does question variation count significantly affect personal fact recall?
2. **RQ2 (Baseline)**: Does fine-tuning compare to simpler methods (RAG, system prompt)?
3. **RQ3 (Depth)**: Does the method scale to complex, multi-hop personal knowledge?
4. **RQ4 (Reproducibility)**: Are results consistent across multiple training runs?

---

## Experiment 1: Ablation Study

### Hypothesis
More question variations per fact will improve recall accuracy, with diminishing returns beyond 20-30 variations.

### Design
| Condition | Variations/Fact | Training Examples | Expected Accuracy |
|-----------|-----------------|-------------------|-------------------|
| Minimal   | 5               | ~25               | 60-70%            |
| Low       | 10              | ~50               | 75-85%            |
| Medium    | 20              | ~100              | 85-95%            |
| High      | 30              | ~150              | 95%+              |

### Controls
- Same base model (Qwen2.5-7B-Instruct)
- Same hyperparameters (r=64, lr=3e-4, epochs=5)
- Same user profile facts
- Same held-out test set
- 3 runs per condition (different random seeds)

### Metrics
- Overall accuracy (% of test questions answered correctly)
- Accuracy by question type (formal, casual, typo, minimal, indirect)
- Training time and loss curves

### Cost
12 runs × $3.0 = **$33.12**

---

## Experiment 2: Baseline Comparison

### Hypothesis
Fine-tuning will compare to baselines, especially on unusual phrasings (typos, minimal, indirect).

### Methods Compared

| Method | Description | Setup Cost | Per-Query Cost |
|--------|-------------|------------|----------------|
| System Prompt | Facts in system message | $0 | Standard |
| RAG | Facts in vector DB, retrieved | ~$0.10 | +Embedding |
| Few-Shot | Example Q&A in context | $0 | +Tokens |
| **Fine-Tune** | QLoRA on 30 variations | $3.0 | Standard |

### Evaluation
Same test set for all methods:
- 20 held-out questions
- 5 difficulty levels
- 5 variation types

### Expected Results
```
                    Formal  Casual  Typo   Minimal  Indirect
System Prompt       90%     70%     40%    30%      40%
RAG                 85%     65%     35%    25%      35%
Few-Shot            90%     80%     50%    40%      50%
Fine-Tune (ours)    95%     95%     90%    85%      75%
```

### Cost
Only fine-tuning has training cost. Evaluation cost negligible.
**~$33** (if training fine-tune baseline 3x each)

---

## Experiment 3: Depth of Knowledge

### Hypothesis
The method works for simple facts but may degrade for complex, multi-hop knowledge.

### Knowledge Tiers

| Tier | Complexity | Example Facts | Count |
|------|------------|---------------|-------|
| 1    | Simple     | Name, age, pet name | ~8 |
| 2    | Relational | Partner, friends, preferences | ~12 |
| 3    | Temporal   | Events, dates, career history | ~10 |
| 4    | Multi-hop  | Partner's birthday, how we met | ~4 |

### Design
Train and evaluate on progressively more complex profiles:
- Tier 1 only (~24 examples)
- Tiers 1-2 (~64 examples)
- Tiers 1-3 (~94 examples)
- Tiers 1-4 (~100 examples)

### Metrics
- Accuracy per tier
- Accuracy degradation as complexity increases
- Multi-hop reasoning success rate

### Cost
12 runs × $3.50 = **$42**

---

## Experiment 4: Reproducibility

### Requirement
Each condition must be run 3 times with different random seeds to establish:
- Mean accuracy
- Standard deviation
- Confidence intervals

### Statistical Tests
- Paired t-tests between conditions
- Effect size (Cohen's d)
- 95% confidence intervals

---

## Held-Out Test Set

### Construction
Test questions are **never seen during training**. They use different phrasings from training variations.

### Categories
| Category | Count | Examples |
|----------|-------|----------|
| Pet questions | 8 | "Can you tell me my pet's name?" |
| Age questions | 5 | "yo how old" |
| Birthday questions | 4 | "bday?" |
| Combined | 3 | "Give me a quick summary of who I am" |
| Negative (shouldn't know) | 2 | "What car do I drive?" |
| **Total** | 22 | |

### Difficulty Levels
- Easy: Direct, formal questions
- Medium: Casual or minimal phrasing
- Hard: Typos, indirect, adversarial

### Evaluation
- Automatic: Check if expected fact appears in response
- Manual: Review ambiguous cases

---

## Cost Summary

| Experiment | Runs | Cost/Run | Total |
|------------|------|----------|-------|
| Ablation | 12 | $3.0 | $33 |
| Baselines | 12 | $3.0 | $33 |
| Depth | 12 | $3.50 | $42 |
| Buffer | - | - | $52 |
| **Total** | 36 | - | **$160** |

---

## Expected Deliverables

After running experiments:

1. **Ablation Results**
   - Table: Accuracy by variation count
   - Figure: Accuracy curve (diminishing returns)
   - Informal comparison tests

2. **Baseline Comparison**
   - Table: Method × Question Type accuracy matrix
   - Figure: Radar chart of method strengths
   - When fine-tuning beats/loses to alternatives

3. **Depth Analysis**
   - Table: Accuracy by knowledge tier
   - Figure: Degradation curve
   - Failure case analysis

4. **Reproducibility**
   - All raw results in JSON
   - Mean ± std for every condition
   - Confidence intervals

---

## Files

```
evaluation/
├── eval_framework.py      # Core evaluation logic
├── ablation_study.py      # Ablation experiment
├── baseline_rag.py        # Baseline comparisons
├── depth_experiment.py    # Depth of knowledge test
├── run_experiments.py     # Main experiment runner
├── METHODOLOGY.md         # This file
└── results/               # Output directory
```

---

## Running the Experiments

```bash
# Dry run (no GPU cost)
python run_experiments.py

# Full run (requires GPU, costs $99-160)
# Edit run_experiments.py: set dry_run=False
python run_experiments.py
```

---

## Ethical Considerations

- All personal data is synthetic (no real PII)
- Training scripts use placeholder values
- Users must substitute their own data
- No data is transmitted externally
