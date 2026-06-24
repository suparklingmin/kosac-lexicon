# Installation

The core install needs only `pandas` and `numpy` — no Java.

```bash
pip install kosac-lexicon                  # core: lexicon data + query API
pip install "kosac-lexicon[kiwi]"          # + Kiwi POS tokenizer (pure pip, no Java)
pip install "kosac-lexicon[transformers]"  # + HuggingFace subword tokenizer
pip install "kosac-lexicon[sklearn]"       # + scikit-learn feature extractor
pip install "kosac-lexicon[all]"           # everything
```

````{note}
During the beta the package is published to
[TestPyPI](https://test.pypi.org/project/kosac-lexicon/), not the main PyPI yet.
Until the 1.0 PyPI release, install from TestPyPI — the extra index pulls the
dependencies (`pandas`, `numpy`, `kiwipiepy`) from the main PyPI:

```bash
pip install --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/ "kosac-lexicon[kiwi]"
```
````

The import name is `kosac`:

```python
import kosac
print(kosac.__version__, kosac.FEATURES)
```

Loading and querying the lexicon works with the core install alone. Add `[kiwi]`
only when you need to tokenize raw sentences (see the
[user guide](manual.md) for end-to-end usage).
