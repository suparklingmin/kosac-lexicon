# High-accuracy polarity classification

The frozen `polarity` lexicon is great for inspecting morpheme-level sentiment,
but for **classifying whole texts as positive or negative** — especially text
from a different domain than KOSAC — two additions in v0.5.0 do much better:
**multi-scale scoring** (the default) and the derived **`polarity-blend`**
lexicon.

## The quick path

```python
from kosac import SentimentAnalyzer

clf = SentimentAnalyzer("polarity-blend")     # multi-scale scoring by default

clf.predict_polarity("이 영화 정말 재미있고 최고였다")   # 'POS'
clf.predict_polarity("시간 낭비 최악의 영화 너무 지루하다")  # 'NEG'
clf.predict_polarity_batch(["좋았다", "별로였다"])        # ['POS', 'NEG']
```

For a continuous score (e.g. to set your own threshold or rank by sentiment):

```python
clf.polarity_score("이 영화 정말 재미있다")   #  0.97   (P(POS) − P(NEG), in [-1, 1])
clf.polarity_score("시간 낭비 최악의 영화")    # -1.00
```

`predict_polarity(text, threshold=0.0)` labels `'POS'` when the score exceeds the
threshold. Raise the threshold to trade recall for precision — useful when the
negative class is rare.

## Why it works

### `polarity-blend`

`polarity-blend` is a derived **POS/NEG** lexicon: the frozen KOSAC seeds combined
with a lexicon learned from the NSMC movie-review corpus. The KOSAC seeds alone
are near-chance out of domain; blending in corpus-learned signal makes it transfer
(on the NIKL out-of-domain benchmark, balanced accuracy rises from ≈0.53 to ≈0.73).

It is loadable by name but is intentionally **not** one of the six canonical
`kosac.FEATURES`, so `SentimentAnalyzer("all")` is unaffected.

```python
import kosac
lex = kosac.load_lexicon("polarity-blend", ngrams=[1, 2, 3])
lex.get_labels()    # ['NEG', 'POS']
```

### Multi-scale scoring

The legacy matcher was greedy leftmost-longest and **non-overlapping**: when a
trigram matched, its unigrams were skipped. Multi-scale scoring instead sums
**every overlapping n-gram** match, so short and long evidence both count. It is
the default; the old behavior is one argument away:

```python
SentimentAnalyzer("polarity-blend", scoring="greedy")             # legacy matcher
SentimentAnalyzer("polarity-blend", ngram_weights={1: 1, 2: 1, 3: 1})  # weight scales
```

(`count()` always stays non-overlapping — double-counting would distort word
tallies. Multi-scale applies to `analyze()` / `polarity_score()`.)

## Regenerating or strengthening the blend

The shipped blend is a compact version. Rebuild it — or build the larger
"champion" variant that scores a little higher — from source (needs the `[kiwi]`
extra; downloads NSMC, CC0):

```bash
python -m benchmarks.build_shipped_blend          # the shipped (compact) blend
python -m benchmarks.build_shipped_blend --full   # the larger champion blend
```

The full benchmark campaign — datasets, decision rules, and every ablation — is
documented in `benchmarks/README.md`.

## License note

`polarity-blend` is derived data under **CC BY-SA 4.0** (it mixes the CC BY-SA
KOSAC seeds with the CC0 NSMC corpus). See `src/kosac/data/polarity-blend.NOTICE`.

## Next steps

- [Sentiment features for machine learning](machine-learning.md)
- [Negation, intensifiers, and tuning](negation-intensifier.md)
- [NSMC case study](nsmc-case-study.md)
