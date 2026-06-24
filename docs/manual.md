# KOSAC Lexicon — User Manual

`kosac-lexicon` packages a **morpheme-level Korean sentiment lexicon** derived
from KOSAC (the Korean Sentiment Analysis Corpus). The distribution name is
`kosac-lexicon`; the import name is `kosac`.

> In a hurry? Jump to [§4 Quick start](#4-quick-start). For the recommended
> end-to-end API see [§6 SentimentAnalyzer](#6-sentimentanalyzer-high-level-api).
> (한국어 매뉴얼: [`manual.ko.md`](manual.ko.md).)

## Table of contents
1. [Introduction](#1-introduction)
2. [Installation](#2-installation)
3. [Core concepts](#3-core-concepts)
4. [Quick start](#4-quick-start)
5. [Working with lexicons (low-level API)](#5-working-with-lexicons-low-level-api)
6. [SentimentAnalyzer (high-level API)](#6-sentimentanalyzer-high-level-api)
7. [Tokenizers](#7-tokenizers)
8. [Command-line interface](#8-command-line-interface)
9. [scikit-learn integration](#9-scikit-learn-integration)
10. [Utilities](#10-utilities)
11. [FAQ & caveats](#11-faq--caveats)
12. [Citation & license](#12-citation--license)

---

## 1. Introduction

The lexicon was built on the assumption that a word's sentiment derives from the
core subjective expressions ("Seeds") that contain it: morpheme N-grams extracted
from KOSAC's Seeds are the lexicon entries. The data was released in 2016 and is
fixed (data version `2016`).

A single text flows through the pipeline like this:

```
Korean sentence → (POS tagging) surface/POS tokens → (lexicon matching)
sentiment expressions → label probability distribution
```

## 2. Installation

```bash
pip install kosac-lexicon                  # core: lexicon data + query API (pandas/numpy only)
pip install "kosac-lexicon[kiwi]"          # + Kiwi POS tokenizer (pure pip, no Java)
pip install "kosac-lexicon[transformers]"  # + HuggingFace subword tokenizer
pip install "kosac-lexicon[sklearn]"       # + scikit-learn feature extractor
pip install "kosac-lexicon[all]"           # everything
```

- **No Java (JVM) is required.** POS tagging uses Kiwi (`kiwipiepy`), which
  installs from PyPI.
- The core install alone loads, queries, and filters the lexicon data. Add
  `[kiwi]` only when you need to tokenize raw sentences.

## 3. Core concepts

### 3.1 Entry format

An entry is an N-gram of `surface/POS` tokens joined by spaces (Sejong tagset):

- unigram: `좋/VA`, `힘/NNG`
- bigram: `행복/NNG 하/XSA`

Each lexicon has 16,361 entries (3,476 unigrams · 6,579 bigrams · 6,307 trigrams).

### 3.2 The six semantic features

List them with `kosac.FEATURES`.

| Feature | Label values |
| --- | --- |
| `polarity` | COMP, NEG, NEUT, None, POS |
| `intensity` | High, Low, Medium, None |
| `expressive-type` | dir-action, dir-explicit, dir-speech, indirect, writing-device |
| `nested-order` | 0, 1, 2, 3 |
| `subjectivity-polarity` | COMP, NEG, NEUT, POS |
| `subjectivity-type` | Agreement, Argument, Emotion, Intention, Judgment, Others, Speculation |

`kosac.describe_feature("polarity")` summarises each label value ([§10](#10-utilities)).

### 3.3 Lexicon data structure

A lexicon is a pandas DataFrame indexed by `entry` (space-joined morphemes), with:

- (per-label columns): the **absolute count** of Seeds with each label value —
  the only data stored in the CSV
- `freq`: number of Seeds (= the row total), derived by the loader
- `max.value` / `max.prop`: the most frequent label and its proportion, derived
  by the loader from the counts

## 4. Quick start

```python
import kosac

# (1) Query the lexicon only — no tokenizer / Java needed
lex = kosac.load_lexicon("polarity", ngrams=[1], min_freq=5)
lex.get_entry("힘/NNG")

# (2) Analyze a sentence — needs Kiwi: pip install "kosac-lexicon[kiwi]"
from kosac import SentimentAnalyzer

analyzer = SentimentAnalyzer("polarity")
result = analyzer.analyze("이 영화 정말 좋다")
result["features"]["polarity"]["label"]   # 'POS'
```

## 5. Working with lexicons (low-level API)

### 5.1 Loading

```python
import kosac
from kosac.lexicon import PolarityLexicon

lex = kosac.load_lexicon("polarity", ngrams=[1, 2, 3], min_freq=0, threshold=0.0)
lex = PolarityLexicon.load(ngrams=[1])                 # equivalent classmethod
lex = PolarityLexicon(filepath="my-lexicon.csv")       # your own CSV
```

- `ngrams`: which N-gram lengths to keep (default `[1]`).
- `feature` in `load_lexicon` must be one of `kosac.FEATURES`.

### 5.2 Filtering & querying

```python
lex.set_lexicon(min_freq=5, threshold=0.6)  # keep freq>=5 AND max.prop>0.6
lex.set_lexicon(min_freq=1)                 # loosen again — re-filters the original
lex.get_size()              # number of entries
lex.get_labels()            # ['COMP','NEG','NEUT','None','POS']
lex.get_lexicon()           # current (filtered) DataFrame
lex.get_original_lexicon()  # the unfiltered original
lex.get_entry("좋/VA")      # one entry's row  (legacy alias: lex.get("좋/VA"))
```

`set_lexicon` always filters the **original** lexicon, so a threshold can be
tightened *or* loosened again.

### 5.3 Matching & scoring (needs a tokenizer)

```python
from kosac.tokenizers import KiwiTokenizer
tok = KiwiTokenizer()

lex.match_patterns("나는 정말 행복하다", tok)   # matched entry strings (alias: lex.match)
lex.get_match_info("나는 정말 행복하다", tok)   # (entry, max.value, max.prop) tuples
lex.get_sent_probs("나는 정말 행복하다", tok)   # per-label probabilities (softmax) Series
```

> `sorting=True` (default) matches longer N-grams / higher-probability entries
> first; `sorting=False` matches in token order.

For richer results (character spans, negation handling, ...) use the
[analyzer](#6-sentimentanalyzer-high-level-api); `match_patterns` / `get_sent_probs`
are plain lexicon lookups.

### 5.4 Building & extending a custom lexicon

```python
from kosac.lexicon import GenericLexicon
from kosac.corpora import Corpus

lex = GenericLexicon(ngrams=[1, 2])
lex.set_labels(["POS", "NEG"])           # GenericLexicon only
lex.add_token("좋/VA", "POS")            # +1 count for a label on an entry
lex.update([("싫/VA", "NEG"), ("좋/VA", "POS")])

corpus = Corpus("data/example.csv")      # headerless text,label CSV
lex.update_from_corpus(corpus, KiwiTokenizer())

lex.export_user_dict("user_dictionary.txt")  # export unigrams as form\tPOS
```

## 6. SentimentAnalyzer (high-level API)

`SentimentAnalyzer` bundles lexicons with a tokenizer and scores text in one
call. **This is the recommended entry point.**

```python
from kosac import SentimentAnalyzer

analyzer = SentimentAnalyzer(
    features="polarity",     # also 'all' or ['polarity','intensity', ...]
    tokenizer=None,          # default KiwiTokenizer()
    ngrams=(1, 2, 3),
    min_freq=0, threshold=0.0,
    smoothing=True,
    align=False,             # seed Kiwi's user dict from the lexicon (§7.3)
    negation=False,          # negation handling (§6.3)
    intensifier=False,       # intensifier handling (§6.3)
    window=2, intensifier_factor=2.0,
)
```

### 6.1 The `analyze(text)` result

```python
r = analyzer.analyze("이 영화 정말 좋다")
```

```python
{
  "text": "이 영화 정말 좋다",
  "tokens": ["이/MM", "영화/NNG", "정말/MAG", "좋/VA", "다/EF"],
  "features": {
    "polarity": {
      "label": "POS",                # top label
      "prob": 0.97,                  # its probability
      "probs": {"COMP": ..., "NEG": ..., "POS": 0.97, ...},
      "matches": [
        {"entry": "좋/VA", "span": [8, 9], "text": "좋",
         "max_value": "POS", "max_prop": 0.92,
         "negated": False, "weight": 1.0},
        ...
      ]
    }
  }
}
```

- `span` is a **character offset** `[start, end)` into the original text, so
  `text[span[0]:span[1]]` recovers the surface string.
- The result is a plain, JSON-serialisable dict.

### 6.2 Multiple features at once / batch

```python
SentimentAnalyzer("all").analyze("이 영화는 정말 좋았고 너무 행복했다")
# polarity=POS, intensity=Medium, expressive-type=dir-speech,
# nested-order=1, subjectivity-polarity=POS, subjectivity-type=Argument

analyzer.analyze_batch(["좋다", "싫다"])     # list of result dicts
analyzer.analyze_frame(["좋다", "싫다"])     # tidy pandas DataFrame
#    text  polarity.label  polarity.prob
# 0   좋다             POS            ...
```

### 6.3 Negation & intensifier

Opt-in heuristics. If a marker appears within `window` tokens of a matched
expression, it is applied.

```python
a = SentimentAnalyzer("polarity", negation=True, intensifier=True)
a.analyze("이 영화는 안 좋다")["features"]["polarity"]["label"]   # 'NEG'
```

- **Negation** (`안/못/않/없 …`): swaps the POS↔NEG mass of the matched
  expression. Applied only to features whose labels contain both POS and NEG
  (`polarity`, `subjectivity-polarity`).
- **Intensifier** (`정말/너무/매우 …`): multiplies the matched expression's weight
  by `intensifier_factor`.
- Marker sets are `kosac.analyzer.DEFAULT_NEGATIONS` / `DEFAULT_INTENSIFIERS` and
  can be replaced via the `negations=` / `intensifiers=` constructor arguments.

```python
from kosac.analyzer import DEFAULT_INTENSIFIERS
SentimentAnalyzer("polarity", intensifier=True,
                  intensifiers=DEFAULT_INTENSIFIERS | {"완전/MAG"}, window=3)
```

> ⚠️ This is a window-based heuristic, so an adjacent morpheme may occasionally
> be flagged as negated. The final label is usually unaffected, but tune
> `window` or post-process if you need precise negation scope.

### 6.4 Frequency-count method

You can classify text by **counting sentiment words** (numbers and proportions of
positive/negative words) instead of using probabilities — the approach common in
content analysis and social-science studies. `count()` provides this: it tallies
matched morphemes by their dominant label (`max.value`).

```python
a = SentimentAnalyzer("polarity")
c = a.count("빗물이 흐르고 내 눈물도 흐르고 잃어버린 첫사랑도 흐르네")["features"]["polarity"]
c["counts"]       # {'NEG': 6, 'POS': 3, ...} — word counts per label
c["proportions"]  # per-label share (sums to 1)
c["total"]        # number of matched words
c["label"]        # most frequent label → 'NEG'

a.count_batch(texts)   # list of result dicts
a.count_frame(texts)   # per-document label counts (<feature>.POS, <feature>.NEG, ...)
```

`analyze()` (probabilities) and `count()` (word counts) aggregate the same
matches differently. Negation/intensifier do not apply to the count method.

## 7. Tokenizers

Module `kosac.tokenizers`.

### 7.1 Kinds

| Class | Description | Extra |
| --- | --- | --- |
| `Tokenizer` | whitespace split (base); for pre-tagged test input | none |
| `KiwiTokenizer` | Kiwi morpheme analysis (recommended); emits `surface/POS` | `[kiwi]` |
| `HuggingFaceTokenizer` | HuggingFace subwords | `[transformers]` |

### 7.2 Common methods

```python
tok.tokenize("나는 좋다")               # ['나/NP', '는/JX', '좋/VA', '다/EF']
tok.get_tokens_str("나는 좋다")         # '나/NP 는/JX 좋/VA 다/EF'
tok.tokenize_with_offsets("나는 좋다")  # [('나/NP', 0, 1), ('는/JX', 1, 2), ...]
tok.get_ngrams("나는 좋다", ns=[1, 2]) # unigram + bigram strings
```

### 7.3 Kiwi user-dictionary alignment

Registering lexicon entries as Kiwi user words makes its segmentation align
better with the lexicon (easing the tagset mismatch).

```python
from kosac.tokenizers import KiwiTokenizer

tok = KiwiTokenizer(user_words=[("행복하", "VA")])   # register directly
tok.add_user_words(["좋/VA", ("멋지", "VA")])         # 'form/tag' or (form, tag)

lex = kosac.load_lexicon("polarity", ngrams=[1])
tok = KiwiTokenizer.from_lexicon(lex, tags={"NNG", "VV", "VA", "XR"})  # seed from lexicon unigrams
```

`SentimentAnalyzer(..., align=True)` applies `from_lexicon` automatically.

## 8. Command-line interface

Installing adds a `kosac` command (or use `python -m kosac`).

```bash
kosac analyze "이 영화 정말 좋다"                 # pretty JSON (probabilities)
kosac analyze "이 영화는 안 좋다" --negation       # negation handling
kosac analyze "좋다" --features all                # all six features
kosac analyze "빗물이 흐르고 눈물도 흐르고" --count  # frequency-count method
kosac features                                     # list features
kosac citation                                     # BibTeX

printf '좋다\n싫다\n' | kosac analyze --compact     # one JSON object per line (JSONL)
```

`analyze` options: `--features` (comma-separated or `all`), `--ngrams`,
`--min-freq`, `--negation`, `--intensifier`, `--align`, `--count`, `--compact`.
With no text argument it reads one sentence per line from stdin.

## 9. scikit-learn integration

`KosacVectorizer` turns text into `<feature>=<label>` probability features
(needs the `[sklearn]` extra).

```python
from kosac.sklearn import KosacVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

clf = make_pipeline(KosacVectorizer("all", negation=True), LogisticRegression())
clf.fit(train_texts, labels)
clf.predict(test_texts)

vec = KosacVectorizer("polarity").fit(train_texts)
vec.get_feature_names_out()   # ['polarity=COMP', 'polarity=NEG', ...]
```

Constructor arguments mirror `SentimentAnalyzer` (`features`, `ngrams`,
`min_freq`, `negation`, `intensifier`, `align`, `tokenizer`).

## 10. Utilities

```python
import kosac

kosac.FEATURES            # tuple of the six feature names
kosac.__version__         # package version (e.g. '1.0.0')
kosac.__data_version__    # data vintage ('2016')

kosac.citation()                    # BibTeX string
kosac.describe_feature("polarity")  # {'feature','values','reference'}
kosac.FEATURE_VALUES["intensity"]   # label value -> description
```

## 11. FAQ & caveats

- **Do I need Java?** No. Kiwi installs from PyPI.
- **Tagset differences.** Entries use the Sejong tagset. Kiwi's tagset is also
  Sejong-based, so the common morpheme tags (NNG/VV/VA/JKS/EC …) match, but some
  symbol/web tags and segmentations may differ from the tagger originally used to
  build KOSAC. Use `align=True` to mitigate.
- **Negation is not perfect.** See the heuristic caveat in [§6.3](#63-negation--intensifier).
- **Wildcard entries.** Some entries contain the regex-special `*` (e.g.
  `가*/JKS`); they are escaped internally and matched literally.
- **No matches.** If nothing matches, a feature's `label` is `None` and `probs`
  is an empty dict (`count` returns `total == 0`).

## 12. Citation & license

```python
print(kosac.citation())
```

- **Code**: MIT (`LICENSE`)
- **Lexicon data** (`kosac/data/*.csv`): CC BY-SA 4.0, derived from KOSAC (Seoul
  National University) (`src/kosac/data/LICENSE`). When redistributing, keep the
  attribution and share-alike terms.
