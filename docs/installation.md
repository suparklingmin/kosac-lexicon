# Installation

The core install needs only `pandas` and `numpy` — no Java.

```bash
pip install kosac-lexicon                  # core: lexicon data + query API
pip install "kosac-lexicon[kiwi]"          # + Kiwi POS tokenizer (pure pip, no Java)
pip install "kosac-lexicon[transformers]"  # + HuggingFace subword tokenizer
pip install "kosac-lexicon[sklearn]"       # + scikit-learn feature extractor
pip install "kosac-lexicon[all]"           # everything
```

The import name is `kosac`:

```python
import kosac
print(kosac.__version__, kosac.FEATURES)
```

Loading and querying the lexicon works with the core install alone. Add `[kiwi]`
only when you need to tokenize raw sentences (see the
[user guide](manual.md) for end-to-end usage).
