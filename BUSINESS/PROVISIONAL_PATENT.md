# Provisional Patent Application Guide

## Andraeus AI Scaling and Context Window Solution

### What is a Provisional Patent?
- 12-month protection while you prepare full patent
- Establishes priority date (December 13, 2025)
- Lower cost than full patent
- Gives you time to assess commercial potential

---

## Step 1: IP Australia Online Filing

**Website:** https://www.ipaustralia.gov.au/patents/applying-patent/provisional-patent-applications

**Cost:** $230 AUD (online) or $460 AUD (paper)

---

## Step 2: Required Information

### Applicant Details
```
Name: Rocco Andraeus Sergi
Address: [YOUR ADDRESS], NSW, Australia
Email: andraeusbeats@gmail.com
```

### Invention Title
```
Method and System for Zero-Context Personal Memory in Large Language
Models Using Weight-Based Knowledge Encoding
```

### Technical Field
```
Artificial Intelligence, Machine Learning, Natural Language Processing,
Large Language Model Optimization
```

---

## Step 3: Provisional Patent Specification

### Abstract (200 words max)
```
A method for encoding personal knowledge directly into large language
model (LLM) weights, eliminating the need for context tokens to store
user-specific information. The invention comprises: (1) a Question
Variation Methodology using 10 optimal variations per fact for robust
knowledge retention; (2) a Tiered Knowledge Architecture with four
complexity levels from simple facts to multi-hop reasoning; (3) a
Zero-Context Personal Memory system that stores facts in model weights
using 0 context tokens; and (4) Scale-Efficient Fine-Tuning techniques
achieving 99% accuracy with 500+ facts. The method solves the AI context
window problem by removing personalization overhead entirely, freeing
100% of context for actual tasks. Implementation uses QLoRA fine-tuning
with LoRA rank 64, alpha 128, targeting all attention and MLP layers.
The method reduces context overhead from 500-5000 tokens to zero tokens
per interaction, providing significant cost savings and improved
performance for personalized AI applications.
```

### Claims (Key Claims)
```
1. A method for encoding personal knowledge into language model weights
   comprising:
   a) Generating multiple question variations for each fact
   b) Training using parameter-efficient fine-tuning
   c) Storing knowledge in weight matrices rather than context

2. The method of claim 1, wherein 10 variations per fact provides
   optimal accuracy of 91.7% or greater.

3. A tiered knowledge architecture comprising:
   a) Tier 1: Simple factual knowledge
   b) Tier 2: Relational knowledge
   c) Tier 3: Temporal knowledge
   d) Tier 4: Multi-hop reasoning knowledge

4. A system achieving zero context token usage for personal information
   while maintaining 99% accuracy at 500 or more facts.

5. The method of claim 1, using LoRA configuration with rank 64 and
   alpha 128 targeting attention and MLP projection layers.
```

### Description (Summary)
```
BACKGROUND
Current AI assistants consume 500-5000 context tokens for personalization,
limiting available context for actual tasks. Existing solutions include
RAG (retrieval), system prompts, and extended context, all of which
consume tokens.

PROBLEM SOLVED
This invention eliminates context overhead by storing personal knowledge
in model weights rather than context tokens.

TECHNICAL SOLUTION
1. Question Variation Methodology: Generate 10 variations per fact
   during training to ensure robust recall regardless of phrasing.

2. Tiered Knowledge Architecture: Organize knowledge into four tiers
   of increasing complexity for systematic encoding.

3. Zero-Context Memory: Use QLoRA fine-tuning to encode facts directly
   into model weights, requiring 0 context tokens at inference.

4. Scale-Efficient Training: Maintain 99% accuracy even with 500+ facts
   through optimized training configuration.

RESULTS
- 99% accuracy at 500 facts
- 0 context tokens for personal information
- $2.76 training cost per user
- 100% context available for tasks
```

---

## Step 4: File Online

1. Go to: https://online.ipaustralia.gov.au
2. Create account or login
3. Select "Patents" → "Provisional Application"
4. Upload specification document
5. Pay $230 AUD fee
6. Receive application number

---

## Step 5: After Filing

| Timeline | Action |
|----------|--------|
| Day 1 | File provisional, get application number |
| Month 1-6 | Market test, seek licensees |
| Month 6-9 | Decide on full patent |
| Month 12 | Must file full patent or PCT or lose priority |

---

## Full Patent Options (within 12 months)

| Option | Coverage | Cost |
|--------|----------|------|
| Australian Patent | Australia only | $3,000-10,000 |
| PCT Application | International | $5,000-15,000 |
| Direct filing | Specific countries | Varies |

---

## DIY vs Patent Attorney

| Approach | Cost | Recommendation |
|----------|------|----------------|
| DIY Provisional | $230 | OK for starting |
| Attorney Provisional | $1,500-3,000 | Better claims |
| Attorney Full Patent | $5,000-15,000 | Recommended |

---

## Resources

- IP Australia: https://www.ipaustralia.gov.au
- Patent application: https://online.ipaustralia.gov.au
- Free patent search: https://search.ipaustralia.gov.au/patents
