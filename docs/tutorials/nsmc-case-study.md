# Case study: a lexicon from NSMC

The bundled lexicons are frozen 2016 KOSAC data, but the
[custom-lexicon](custom-lexicon.md) machinery builds a lexicon from **any**
labeled corpus. This walks through a realistic end-to-end build on
[NSMC](https://github.com/e9t/nsmc) (Naver Sentiment Movie Corpus — 200k movie
reviews labeled positive / negative), the same thing
[`examples/nsmc_lexicon.py`](https://github.com/suparklingmin/kosac-lexicon/blob/main/examples/nsmc_lexicon.py)
does as a script.

## The corpus

NSMC ships as `id<TAB>document<TAB>label` (label `1` = positive, `0` = negative).
`Corpus` reads a headerless `text,label` CSV, so map the labels to the lexicon's
and drop blank rows:

```python
import pandas as pd

df = pd.read_csv('ratings_train.txt', sep='\t', quoting=3, keep_default_na=False)
df['document'] = df['document'].astype(str).str.strip()
df = df[df['document'] != '']
df['label'] = df['label'].map({1: 'POS', 0: 'NEG'})
df[['document', 'label']].to_csv('nsmc_train.csv', index=False, header=False)
```

## Building

`update_from_corpus` tokenizes each review into morpheme N-grams and counts its
label for every N-gram. Restrict to **content words** with `pos_tag` so the
result is meaning-bearing words rather than `다/EF`, `가/JKS`, …; drop rare
entries with `min_freq`:

```python
from kosac.corpora import Corpus
from kosac.lexicon import GenericLexicon
from kosac.tokenizers import KiwiTokenizer

corpus = Corpus('nsmc_train.csv')
lex = GenericLexicon(ngrams=[1, 2, 3])
lex.set_labels(['POS', 'NEG'])
lex.update_from_corpus(
    corpus, KiwiTokenizer(),
    pos_tag={'NNG', 'NNP', 'VV', 'VA', 'XR', 'MAG'},   # content words
    min_freq=5,
)
lex.get_size()        # ~104k entries from the full 150k-review train set
```

```{note}
The build counts ~1.6M distinct N-grams over 150k reviews. Tokenization runs in
one Kiwi batch and counting is vectorized, so the whole train set builds in about
15 seconds; the `min_freq` filter then trims it to the entries seen ≥ 5 times.
```

## Inspecting

The top content unigrams by dominant-label proportion (`max.prop`) are a good
sanity check — they line up with movie-review sentiment:

```python
df = lex.get_lexicon()
uni = df[(df['ngram'] == 1) & (df['freq'] >= 100)]
uni[uni['max.value'] == 'POS'].sort_values('max.prop', ascending=False).head(10)
```

| POS | NEG |
|-----|-----|
| 먹먹/펑펑/뭉클/아련 (moving) | 최악 (worst) |
| 여운 (lingering impression) | 노잼 (not funny) |
| 따뜻 (warm), 최고 (the best) | 낭비 (waste), 지루 (boring) |
| 유쾌 (pleasant), 수작 (fine work) | 졸작 (botch), 쓰레기 (trash), 짜증 (annoyance) |

## Scoring

A custom lexicon supports the same matching and scoring as the bundled ones:

```python
tok = KiwiTokenizer()
lex.set_lexicon(min_freq=50)        # shrink the match regex for interactive use
lex.get_sent_probs('이 영화 정말 최고였다 여운이 남는다', tok).round(3).to_dict()
# {'POS': 0.999, 'NEG': 0.001}
lex.get_sent_probs('시간 낭비 최악의 영화 너무 지루하다', tok).round(3).to_dict()
# {'NEG': 1.0, 'POS': 0.0}
```

## How good is it?

Because NSMC ships a 50k test split, you can measure the lexicon directly. Using
a greedy leftmost-longest, non-overlapping matcher and predicting the label with
the larger summed (add-one smoothed) log-probability — **no training, just the
lexicon** — gives, on the test set:

| metric | value |
|--------|-------|
| coverage (≥ 1 match) | 97.4% |
| accuracy (all reviews) | **83.3%** |
| accuracy (covered reviews) | 85.5% |

That is a strong unsupervised baseline (supervised models land around 85–90%).

## Tuning

Sweeping one knob at a time over the full train/test split. *cov* is the fraction
of reviews with ≥ 1 match; **acc. (all)** is accuracy over every test review, while
**acc. (covered)** is accuracy over just the covered ones (so the gap between them
is what coverage costs):

| knob | setting | entries | coverage | acc. (all) | acc. (covered) |
|------|---------|--------:|---------:|-----------:|---------------:|
| **N-gram order** | `[1]` | 12k | 97.4% | 80.6% | 82.7% |
| | `[1, 2]` | 57k | 97.4% | 82.4% | 84.6% |
| | `[1, 2, 3]` | 105k | 97.4% | **83.3%** | 85.5% |
| **min_freq** | 3 | 193k | 97.5% | 83.2% | 85.4% |
| | 5 | 105k | 97.4% | 83.3% | 85.5% |
| | 20 | 24k | 97.0% | 82.5% | 85.0% |
| **max.prop ≥** | 0.0 | 105k | 97.4% | 83.3% | 85.5% |
| | 0.7 | 62k | 94.3% | 81.4% | 86.4% |
| | 0.8 | 47k | 88.6% | 77.7% | **87.7%** |
| **POS filter** | content words | 105k | 97.4% | 83.3% | 85.5% |
| | content + `IC` | 106k | 97.8% | 83.7% | 85.6% |
| | **all tags** | 131k | **99.6%** | **85.9%** | 86.2% |

Takeaways:

- **Higher N-grams help** — bigrams and trigrams add ~3 points over unigrams.
- **`min_freq` mostly controls size, not accuracy** — dropping from 193k to 105k
  entries (`min_freq` 3 → 5) barely moves the score; it is a size/precision dial.
- **`max_value_threshold` trades coverage for precision** — at 0.8 the lexicon is
  more accurate *on the reviews it covers* (87.7%) but covers far fewer, so overall
  accuracy falls.
- **The content-word filter costs accuracy.** Keeping *all* tags — function
  morphemes, punctuation, and emoticons (`ㅋㅋ`, `!!!`, `…`) all carry sentiment in
  short reviews — reaches 85.9% at 99.6% coverage. So pick by goal: filter to
  content words for a **clean, human-readable** lexicon; keep everything to
  **maximize classification accuracy**:

  ```python
  # accuracy-max build: omit pos_tag to keep every token
  lex.update_from_corpus(corpus, KiwiTokenizer(), min_freq=5)   # ~85.9% on NSMC test
  ```

  The example script exposes this as `--pos-filter`:
  `python examples/nsmc_lexicon.py --pos-filter all`.

```{note}
This lexicon isn't bundled with the package — it's something you build from NSMC
with the snippets above (or `examples/nsmc_lexicon.py`). The package ships only
the frozen KOSAC lexicons; `update_from_corpus` is the supported way to derive
your own from any labeled corpus.
```

## Saving

`save()` writes the package's absolute-count CSV (morphemes space-joined), which
the constructor reads back — labels and all, including punctuation morphemes like
`;/SP` whose surface is itself a separator-looking character:

```python
lex.save('nsmc_lexicon.csv')

reloaded = GenericLexicon(filepath='nsmc_lexicon.csv', ngrams=[1, 2, 3])
```
