# Counting sentiment words (social science)

Many social-science studies classify text by **counting sentiment words** — the
number and proportion of positive vs. negative words — rather than by a model's
probability. `kosac` provides this with `count()`.

## Per-document counts

`count()` tallies each matched morpheme by its dominant label (`max.value`):

```python
from kosac import SentimentAnalyzer

analyzer = SentimentAnalyzer("polarity")
c = analyzer.count("서비스가 형편없고 너무 실망스러웠다")["features"]["polarity"]

c["counts"]        # {'COMP': 0, 'NEG': 3, 'NEUT': 0, 'None': 0, 'POS': 2}
c["proportions"]   # {'NEG': 0.6, 'POS': 0.4, ...}
c["total"]         # 5
c["label"]         # 'NEG'  (most frequent label)
```

## Over a corpus

`count_frame()` returns one row per document, with a count column per label:

```python
texts = [
    "이 정책은 정말 훌륭하고 만족스럽다",
    "서비스가 형편없고 너무 실망스러웠다",
    "그저 그런 평범한 하루였다",
]
analyzer.count_frame(texts)
```

```text
                  text polarity.label  polarity.total  polarity.NEG  polarity.POS  ...
0  이 정책은 정말 훌륭하고 만족스럽다            NEG               8             5             3
1  서비스가 형편없고 너무 실망스러웠다            NEG               5             3             2
2       그저 그런 평범한 하루였다           COMP               5             2             0
```

## ⚠️ Caveat: function morphemes add noise

Look at row 0: a clearly **positive** sentence is counted as **NEG**, because
every matched morpheme is counted — including particles and endings (조사, 어미)
whose lexicon polarity is noisy. This is a known limitation of the pure
word-counting approach.

For **per-sentence classification**, prefer the probability method `analyze()`,
which weighs the evidence instead of counting it:

```python
analyzer.analyze_frame(texts)
```

```text
                  text polarity.label  polarity.prob
0  이 정책은 정말 훌륭하고 만족스럽다            POS           0.59
1  서비스가 형편없고 너무 실망스러웠다            NEG           0.66
2       그저 그런 평범한 하루였다            NEG           0.46
```

Now row 0 is correctly **POS**.

## When the count method shines: corpus-level trends

Word counts are most useful as **descriptive statistics over a whole corpus**,
not single sentences. Sum the per-document counts to gauge the overall tone of a
collection:

```python
frame = analyzer.count_frame(texts)
totals = frame[["polarity.POS", "polarity.NEG", "polarity.NEUT"]].sum()
totals.to_dict()
# {'polarity.POS': 5, 'polarity.NEG': 10, 'polarity.NEUT': 1}

positivity = totals["polarity.POS"] / (totals["polarity.POS"] + totals["polarity.NEG"])
round(positivity, 3)   # 0.333
```

With only three sentences here the noise still dominates (`positivity` is 0.33
despite a balanced set) — which is exactly why this method needs a large corpus.
Across thousands of documents the per-morpheme noise averages out and the
positive/negative ratio becomes a meaningful indicator, as used in the literature.

## Reading texts from a file

`count_frame` / `analyze_frame` take any iterable of strings, so a pandas column
works directly:

```python
import pandas as pd

df = pd.DataFrame({"text": texts})         # e.g. pd.read_csv("reviews.csv")
analyzer.count_frame(df["text"]).shape     # (3, 8)
```
