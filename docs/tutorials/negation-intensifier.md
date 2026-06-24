# Negation, intensifiers, and tuning

A plain lexicon lookup gets `안 좋다` ("not good") wrong, because it never sees
the negation. `kosac` offers opt-in composition to fix this.

## The problem

```python
from kosac import SentimentAnalyzer

plain = SentimentAnalyzer("polarity")
plain.analyze("이 영화는 안 좋다")["features"]["polarity"]["label"]   # 'POS'  (wrong!)
```

The positive word `좋` dominates, so the sentence scores POS despite the `안`.

## Turn on negation

```python
negating = SentimentAnalyzer("polarity", negation=True)
negating.analyze("이 영화는 안 좋다")["features"]["polarity"]["label"]   # 'NEG'
```

When a negation marker (`안 / 못 / 않 / 없 …`) falls within a window of a matched
expression, its POS↔NEG mass is swapped:

| sentence | plain | `negation=True` |
| --- | --- | --- |
| `이 영화는 안 좋다` | POS (0.66) | **NEG (0.99)** |
| `별로 좋지 않다` | POS (0.66) | **NEG (0.95)** |

Each affected match is flagged in the result:

```python
for m in negating.analyze("이 영화는 안 좋다")["features"]["polarity"]["matches"]:
    print(m["entry"], m["negated"])
# 이/MM False
# 영화/NNG True
# 는/JX 안/MAG False
# 좋/VA True
# 다/EC True
```

(`영화/NNG` and `다/EC` are also flagged here because they fall inside the window
around `안` — see the caveat below.)

## Intensifiers

Intensifier markers (`정말 / 너무 / 매우 …`) up-weight a nearby match, sharpening
the dominant label:

```python
plain   = SentimentAnalyzer("polarity")
boosted = SentimentAnalyzer("polarity", intensifier=True, intensifier_factor=2.0)

plain.analyze("정말 좋다")["features"]["polarity"]["probs"]["POS"]     # 0.927
boosted.analyze("정말 좋다")["features"]["polarity"]["probs"]["POS"]   # 0.992
```

You can enable both at once: `SentimentAnalyzer("polarity", negation=True,
intensifier=True)`.

## Aligning Kiwi to the lexicon

`align=True` seeds Kiwi's user dictionary with the lexicon's entries, so its
segmentation matches the lexicon's `surface/POS` units more often:

```python
analyzer = SentimentAnalyzer("polarity", align=True)
analyzer.analyze("이 영화 정말 좋다")["features"]["polarity"]["label"]   # 'POS'
```

## Customising the markers

Marker sets, the window size, and the boost factor are all configurable. The
defaults are exported for reuse:

```python
from kosac.analyzer import DEFAULT_NEGATIONS

custom = SentimentAnalyzer(
    "polarity",
    negation=True,
    negations=DEFAULT_NEGATIONS | {"별로/MAG"},   # treat 별로 as a negation marker
    window=3,                                      # tokens to look around a match
)
custom.analyze("이 결과는 별로 좋다")["features"]["polarity"]["label"]   # 'NEG' (0.82)
# without the 별로 marker, the same sentence stays 'POS' (0.53)
```

## ⚠️ Caveat

Negation/intensifier handling is a **window-based heuristic**. With a wide
`window` an adjacent morpheme can occasionally be flagged by mistake; the final
label is usually unaffected, but tune `window` (or post-process the per-match
`negated` flags) if you need precise negation scope.
