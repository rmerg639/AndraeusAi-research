# Provisional Patent Application Guide

## Andraeus AI Question Variation Methodology for Personal AI Fine-Tuning

**IMPORTANT DISCLAIMER:** This guide is for informational purposes only. Patent applications should be prepared with the assistance of a registered patent attorney. The claims below are DRAFT suggestions and require professional review to ensure they are novel, non-obvious, and properly scoped.

---

## Prior Art Acknowledgment

Before filing, be aware of the following prior art that may affect patentability:

| Prior Art | Date | Relevance |
|-----------|------|-----------|
| LoRA: Low-Rank Adaptation (Microsoft) | 2021 | Core fine-tuning technique used |
| QLoRA (Dettmers et al.) | 2023 | Quantization approach used |
| PEFT Library (Hugging Face) | 2022 | Implementation framework |
| Prompt Tuning (Lester et al.) | 2021 | Alternative parameter-efficient method |
| Adapter Modules (Houlsby et al.) | 2019 | Alternative fine-tuning approach |

**Key Differentiation Required:** The invention must focus on the NOVEL aspects, not the underlying LoRA/fine-tuning techniques which are prior art.

---

## What May Be Patentable

The potentially novel aspects are:
1. The specific question variation methodology (10 variations per fact)
2. The tiered knowledge architecture for organizing personal facts
3. The specific training recipe optimized for personal knowledge retention

**What is NOT patentable:**
- Using LoRA for fine-tuning (prior art)
- Using quantization (prior art)
- The general concept of personalizing LLMs (prior art)

---

## Step 1: IP Australia Online Filing

**Website:** https://www.ipaustralia.gov.au/patents/applying-patent/provisional-patent-applications

**Cost:** $230 AUD (online) or $460 AUD (paper)

**Strong Recommendation:** Given the crowded prior art landscape, consult a patent attorney before filing to assess whether the claims are defensible.

---

## Step 2: Required Information

### Applicant Details
```
Name: Rocco Andraeus Sergi
Address: [YOUR ADDRESS], NSW, Australia
Email: andraeusbeats@gmail.com
```

### Invention Title (Narrowed for Novelty)
```
Question Variation Methodology for Robust Personal Knowledge
Encoding in Fine-Tuned Language Models
```

**Note:** Removed broad terms like "Zero-Context" which describe an outcome, not a novel method.

### Technical Field
```
Machine Learning, Natural Language Processing, Language Model
Personalization, Parameter-Efficient Fine-Tuning
```

---

## Step 3: Provisional Patent Specification

### Abstract (200 words max) - REVISED FOR ACCURACY
```
A method for improving personal knowledge retention in fine-tuned
language models through systematic question variation during training
data preparation. The invention addresses the problem of unreliable
recall when users phrase questions differently than training examples.

The method comprises: (1) a Question Variation Methodology that
generates multiple phrasings (optimally 10) for each personal fact,
including formal, casual, abbreviated, and indirect formulations;
(2) a Tiered Knowledge Architecture that organizes personal information
into four complexity levels for systematic encoding; and (3) specific
training hyperparameters optimized for personal fact retention.

In experimental evaluation, the method achieved 91.7% accuracy on
held-out test questions with varied phrasings, compared to lower
accuracy when training with single phrasings per fact. The method
builds upon existing parameter-efficient fine-tuning techniques
(LoRA/QLoRA) and applies them specifically to personal knowledge
encoding with optimized variation strategies.

Note: Accuracy figures are from controlled experiments and may vary
in production use. The underlying fine-tuning techniques (LoRA, QLoRA)
are prior art; the claimed invention is the question variation
methodology and training recipe.
```

### Claims (NARROWED FOR DEFENSIBILITY)

```
INDEPENDENT CLAIMS:

1. A method for preparing training data for personal knowledge
   encoding in language models, comprising:
   a) Receiving a set of personal facts about a user;
   b) For each fact, automatically generating a plurality of
      question-answer pairs with varied phrasings including:
      - Formal question phrasing
      - Casual/colloquial phrasing
      - Abbreviated phrasing
      - Indirect/contextual phrasing
      - Phrasing with common typos or misspellings
   c) Combining the varied question-answer pairs into a training
      dataset for fine-tuning.

2. The method of claim 1, wherein the plurality comprises
   approximately 10 variations per fact, determined experimentally
   to balance training efficiency with recall robustness.

DEPENDENT CLAIMS:

3. The method of claim 1, further comprising organizing the personal
   facts into a tiered architecture comprising:
   a) Tier 1: Simple factual attributes (name, age, location)
   b) Tier 2: Relational facts (pet names, family members)
   c) Tier 3: Temporal facts (birthdays, anniversaries)
   d) Tier 4: Complex multi-hop facts requiring inference

4. The method of claim 1, wherein the training uses parameter-
   efficient fine-tuning with:
   - LoRA rank of 64
   - LoRA alpha of 128
   - Learning rate of 3e-4
   - 5 training epochs
   as optimized for personal knowledge retention.

5. A computer-implemented system for executing the method of claim 1,
   comprising:
   a) A variation generator module that produces question phrasings
   b) A training data formatter that structures examples for fine-tuning
   c) An integration layer for parameter-efficient fine-tuning libraries

6. The method of claim 1, wherein generating varied phrasings includes
   programmatically applying transformations comprising:
   - Removing punctuation
   - Converting to lowercase
   - Substituting synonyms
   - Restructuring as statements rather than questions
```

**CLAIMS INTENTIONALLY OMITTED (Prior Art):**
- Claims about "zero context tokens" - this is an outcome, not a method
- Claims about LoRA/QLoRA techniques - prior art
- Claims about "99% accuracy" - performance claims are not patentable methods
- Claims about cost savings - business outcomes, not technical methods

---

### Description (REVISED FOR ACCURACY)

```
BACKGROUND

Large language models (LLMs) can be personalized through various
methods including system prompts, retrieval-augmented generation
(RAG), and fine-tuning. Fine-tuning offers the advantage of encoding
knowledge directly in model weights, but prior approaches suffer from
poor recall when users phrase questions differently than training
examples.

PROBLEM ADDRESSED

When fine-tuning LLMs on personal facts with single question-answer
pairs, the model often fails to recall facts when users ask questions
with different phrasing, typos, or indirect references. This limits
the practical utility of personal fine-tuning.

TECHNICAL CONTRIBUTION

This invention provides a systematic methodology for generating
training data that improves recall robustness:

1. Question Variation Methodology:
   For each personal fact, generate multiple question phrasings:
   - "What is my dog's name?" (formal)
   - "whats my dogs name" (casual, no punctuation)
   - "my dog" (minimal)
   - "Do you know my pet?" (indirect)
   - "wat is my dogs name" (common typo)

   Experimental testing found that 10 variations per fact provides
   optimal balance between training efficiency and recall robustness.
   Fewer variations (5) showed reduced robustness; more variations
   (20-30) showed diminishing returns.

2. Tiered Knowledge Architecture:
   Organize personal facts by complexity to ensure systematic coverage:
   - Tier 1: Direct attributes (name, age, location)
   - Tier 2: Relational knowledge (pet name, family)
   - Tier 3: Temporal knowledge (dates, events)
   - Tier 4: Multi-hop (requires combining facts)

3. Optimized Training Recipe:
   Through experimentation, the following hyperparameters were found
   effective for personal knowledge retention:
   - LoRA rank: 64 (balances capacity and efficiency)
   - LoRA alpha: 128 (2x rank scaling)
   - Learning rate: 3e-4 (higher than typical fine-tuning)
   - Epochs: 5 (sufficient for memorization without overfitting)

EXPERIMENTAL RESULTS

In controlled experiments with 100 personal facts:
- Single phrasing training: 67% accuracy on varied test questions
- 5 variations per fact: 82% accuracy
- 10 variations per fact: 91.7% accuracy
- 20 variations per fact: 92.1% accuracy (diminishing returns)

Note: These results are from specific experimental conditions.
Real-world performance may vary based on fact complexity, model
choice, and evaluation methodology.

LIMITATIONS AND SCOPE

This invention applies specifically to the training data preparation
phase. The underlying fine-tuning techniques (LoRA, QLoRA, etc.) are
existing methods not claimed as part of this invention.

The method is most applicable to factual personal knowledge and may
be less effective for:
- Complex reasoning tasks
- Rapidly changing information
- Knowledge requiring external verification
```

---

## Step 4: Prior Art Search (Required Before Filing)

Conduct searches on:
- Google Patents: https://patents.google.com
- IP Australia: https://search.ipaustralia.gov.au/patents
- USPTO: https://www.uspto.gov/patents/search

Search terms:
- "question variation language model"
- "personal knowledge fine-tuning"
- "training data augmentation NLP"
- "phrasing robustness neural network"

Document any relevant prior art found and ensure claims differentiate from it.

---

## Step 5: File Online

1. Go to: https://online.ipaustralia.gov.au
2. Create account or login
3. Select "Patents" → "Provisional Application"
4. Upload specification document
5. Pay $230 AUD fee
6. Receive application number

---

## Step 6: After Filing

| Timeline | Action |
|----------|--------|
| Day 1 | File provisional, get application number |
| Month 1-3 | Conduct thorough prior art search |
| Month 3-6 | Consult patent attorney on claim scope |
| Month 6-9 | Assess commercial potential |
| Month 12 | Decide: file full patent, PCT, or abandon |

---

## Cost-Benefit Analysis

| Scenario | Recommendation |
|----------|---------------|
| Strong commercial interest | File PCT with attorney assistance |
| Moderate interest | File Australian patent with attorney |
| Uncertain market | Let provisional lapse, publish as prior art |
| No commercial interest | Don't file, focus on open-source |

**Reality Check:** Patents are expensive ($5,000-50,000+ for full prosecution) and only valuable if you can afford to enforce them. For solo inventors, publishing research openly may provide more practical benefit than a patent you cannot afford to defend.

---

## Professional Resources

- IP Australia: https://www.ipaustralia.gov.au
- Find a Patent Attorney: https://www.ipaustralia.gov.au/tools-resources/find-attorneys
- Free IP Consultation: https://www.ipaustralia.gov.au/about-us/contact-us

**Strong Recommendation:** Before spending money on patents, consult with a patent attorney for a freedom-to-operate opinion and patentability assessment.

---

**END OF GUIDE**
