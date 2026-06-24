# Getting started

Your first sentiment analysis with `kosac`, end to end.

## Install

```bash
pip install "kosac-lexicon[kiwi]"
```

The `[kiwi]` extra adds the Kiwi morpheme tokenizer — it installs from PyPI with
no Java required.

## Analyze a sentence

```python
from kosac import SentimentAnalyzer

analyzer = SentimentAnalyzer("polarity")
result = analyzer.analyze("이 영화 정말 좋다")

result["features"]["polarity"]["label"]   # 'POS'
round(result["features"]["polarity"]["prob"], 3)   # 0.968
```

## Read the result

`analyze()` returns a plain, JSON-serialisable dict. Each feature carries the top
`label`, its `prob`, the full `probs` distribution, and the matched entries —
each with a **character span** back into the original text:

```python
for m in result["features"]["polarity"]["matches"]:
    print(f'{m["entry"]:>8}  {m["span"]}  {m["text"]!r}  {m["max_value"]}')
```

```text
    이/MM  [0, 1]  '이'   NEG
  영화/NNG  [2, 4]  '영화'  POS
  정말/MAG  [5, 7]  '정말'  POS
    좋/VA  [8, 9]  '좋'   POS
    다/EF  [9, 10]  '다'  NEG
```

Notice that function morphemes such as `이/MM` and `다/EF` carry noisy polarity
(their corpus distribution leans NEG). The probability aggregation weighs all the
evidence — dominated here by `좋/VA` (POS) — and still yields **POS** overall.

`span` lets you map a match back to the source text:

```python
m = result["features"]["polarity"]["matches"][3]
result["text"][m["span"][0]:m["span"][1]]   # '좋'
```

## All six features at once

KOSAC annotates six independent features. Ask for them together with
`features="all"`:

```python
SentimentAnalyzer("all").analyze("이 영화는 정말 좋았고 너무 행복했다")
```

| feature | label | prob |
| --- | --- | --- |
| polarity | POS | 0.98 |
| intensity | Medium | 1.00 |
| expressive-type | dir-speech | 1.00 |
| nested-order | 1 | 1.00 |
| subjectivity-polarity | POS | 1.00 |
| subjectivity-type | Argument | 0.98 |

## Next steps

- [Counting sentiment words (social science)](social-science.md)
- [Sentiment features for machine learning](machine-learning.md)
- [Negation, intensifiers, and tuning](negation-intensifier.md)
