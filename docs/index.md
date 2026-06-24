# kosac-lexicon

A morpheme-level **Korean sentiment lexicon** derived from KOSAC (the Korean
Sentiment Analysis Corpus), packaged for Python. The import name is `kosac`; the
distribution name is `kosac-lexicon`.

```{note}
Pre-release (beta, 0.2.0). The API may change before the 1.0 release.
```

```python
import kosac

lex = kosac.load_lexicon("polarity", ngrams=[1], min_freq=5)
lex.get_entry("힘/NNG")

analyzer = kosac.SentimentAnalyzer("polarity", negation=True)   # needs [kiwi]
analyzer.analyze("이 영화는 안 좋다")["features"]["polarity"]["label"]   # 'NEG'
```

```{toctree}
:maxdepth: 1
:caption: Tutorials

tutorials/getting-started
tutorials/counting
tutorials/machine-learning
tutorials/negation-intensifier
tutorials/custom-lexicon
tutorials/nsmc-case-study
```

```{toctree}
:maxdepth: 2
:caption: Guide

installation
manual
api
changelog
```

```{toctree}
:maxdepth: 2
:caption: 한국어 (Korean)

manual.ko
```
