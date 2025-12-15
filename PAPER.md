# Notes on Question Variation Experiments

Personal experimentation notes. **NOT peer-reviewed. NOT validated.**

---

## Idea

Generate multiple question phrasings per fact when fine-tuning.

## Approach

```
Fact: Dog's name is Buddy
Variations: "What's my dog's name?", "my dog?", "whats my dogs name"
```

## Observations

In informal tests:
- More variations seemed to help recall
- Results varied between runs
- RAG/system prompts worked better in my tests

## Limitations

- Self-designed tests
- Single model
- No replication
- Substring matching evaluation

## Conclusion

May help in some cases. RAG/system prompts are simpler alternatives.

---

**EXPERIMENTAL. USE AT OWN RISK.**
