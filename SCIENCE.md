# Technical Notes

Personal notes on fine-tuning experiments.

**NOT peer-reviewed. NOT validated. Personal experimentation only.**

---

## The Idea

Train with multiple question phrasings per fact to improve recall robustness.

---

## Why It Might Help

When you fine-tune on a single phrasing:
- Model learns narrow pattern
- Different phrasings may fail

With multiple phrasings:
- Model may learn underlying fact
- More robust to variation

---

## Technical Details

- QLoRA: 4-bit quantization + LoRA adapters
- LoRA rank 64, alpha 128
- Small adapter size (~1.5MB)

---

## Limitations

- Only tested on one model
- Small datasets
- Self-designed evaluation
- No independent replication
- Results vary between runs

---

## Alternatives

- **RAG**: Better for dynamic info, traceable sources
- **System prompts**: Simpler, no training needed

These may work better for your use case.

---

**EXPERIMENTAL. USE AT OWN RISK.**
