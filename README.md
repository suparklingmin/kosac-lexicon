# kosac-lexicon

A morpheme-level **Korean sentiment lexicon** derived from the Korean Sentiment
Analysis Corpus (**KOSAC**), packaged for easy reuse in Python.

The lexicon maps Korean morpheme N-grams (`surface/POS`, Sejong tagset) to
distributions over six independent sentiment features. It was constructed from
KOSAC's annotated sentiment "Seed" expressions and first released as CSV files
in 2016 (see [`README.ko.md`](README.ko.md) and
[`src/kosac/data/readme.txt`](src/kosac/data/readme.txt) for the linguistic
derivation, in Korean).

## Installation

```bash
pip install kosac-lexicon                  # core: lexicon data + query API (pandas/numpy only)
pip install "kosac-lexicon[kiwi]"          # + Kiwi POS tokenizer (pure pip, no Java needed)
pip install "kosac-lexicon[transformers]"  # + HuggingFace subword tokenizer
pip install "kosac-lexicon[sklearn]"       # + scikit-learn feature extractor
pip install "kosac-lexicon[all]"           # everything
```

The import name is `kosac`:

```python
import kosac
```

## Quickstart

Loading and querying a lexicon needs no file paths and no Java — the data ships
with the package:

```python
import kosac

kosac.FEATURES
# ('polarity', 'intensity', 'expressive-type', 'nested-order',
#  'subjectivity-polarity', 'subjectivity-type')

lex = kosac.load_lexicon('polarity', ngrams=[1], min_freq=5)
lex.get_size()
lex.get_entry('힘/NNG')          # sentiment distribution for a single morpheme
```

To tag raw sentences and score sentiment you need a tokenizer (an optional
extra). The recommended one is Kiwi, which installs from PyPI with no Java:

```python
import kosac
from kosac.tokenizers import KiwiTokenizer   # pip install "kosac-lexicon[kiwi]"

tok = KiwiTokenizer()
lex = kosac.load_lexicon('polarity', ngrams=[1, 2, 3])

lex.match_patterns('나는 정말 행복하다', tok)   # matched sentiment entries
lex.get_sent_probs('나는 정말 행복하다', tok)   # P(label) over the sentence
```

The lexicon uses Sejong-style `surface/POS` entries. Kiwi's tagset is
Sejong-based and aligns on the common morpheme tags, so it matches the lexicon
well; some symbol/web tags differ and may segment slightly differently from the
tagger originally used to build KOSAC.

You can also use the per-feature classes directly, or pass your own CSV:

```python
from kosac.lexicon import PolarityLexicon

PolarityLexicon.load(ngrams=[1, 2])             # bundled data
PolarityLexicon(filepath='my-lexicon.csv')      # custom file
```

## High-level analysis

`SentimentAnalyzer` bundles the lexicons with a tokenizer (Kiwi by default) and
scores text in one call — optionally across all six features at once, with
opt-in negation/intensifier handling:

```python
from kosac import SentimentAnalyzer

analyzer = SentimentAnalyzer('all', negation=True, intensifier=True)
result = analyzer.analyze('이 영화는 정말 좋았다')
result['features']['polarity']['label']      # 'POS'
result['features']['polarity']['matches']    # matched entries with character spans

# negation is handled:
neg = SentimentAnalyzer('polarity', negation=True)
neg.analyze('이 영화는 안 좋다')['features']['polarity']['label']   # 'NEG'

analyzer.analyze_frame(['좋다', '싫다'])       # tidy pandas DataFrame

# frequency-count method (counts of POS/NEG/... words), common in social science:
analyzer.count('빗물이 흐르고 눈물도 흐르고')['features']['polarity']['counts']
```

`align=True` seeds Kiwi's user dictionary from the lexicon to reduce the tagset
mismatch. Negation/intensifier are window-based heuristics and off by default.
`count()` / `count_batch()` / `count_frame()` provide the word-count method
instead of probabilities.

### Command line

```bash
kosac analyze "이 영화 정말 좋다" --features all --negation
kosac features
kosac citation
printf '좋다\n싫다\n' | kosac analyze --compact
```

### scikit-learn

```python
from kosac.sklearn import KosacVectorizer          # pip install "kosac-lexicon[sklearn]"
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

clf = make_pipeline(KosacVectorizer('all'), LogisticRegression())
clf.fit(train_texts, labels)
```

`kosac.citation()` returns BibTeX, and `kosac.describe_feature('polarity')`
summarises a feature's label values.

## Sentiment features

| Feature (`kosac.FEATURES`) | Label values |
| --- | --- |
| `polarity` | COMP, NEG, NEUT, None, POS |
| `intensity` | High, Low, Medium, None |
| `expressive-type` | dir-action, dir-explicit, dir-speech, indirect, writing-device |
| `nested-order` | 0, 1, 2, 3 |
| `subjectivity-polarity` | COMP, NEG, NEUT, POS |
| `subjectivity-type` | Agreement, Argument, Emotion, Intention, Judgment, Others, Speculation |

Each lexicon has 16,361 entries (3,476 unigrams, 6,579 bigrams, 6,307 trigrams).
For each N-gram: `freq` (number of Seeds containing it), one column per label
(relative frequency, converted to absolute counts on load), `max.value` (most
frequent label), and `max.prop` (its proportion).

## Documentation

The full user manual (Korean) is at [`docs/manual.ko.md`](docs/manual.ko.md) —
it covers every part of the API with examples.

## Examples

See [`examples/quickstart.py`](examples/quickstart.py) for a runnable script and
[`examples/`](examples/) for notebooks. The original 2024 prototypes live in
[`examples/legacy/`](examples/legacy/).

## License

- **Code** — MIT (see [`LICENSE`](LICENSE)).
- **Lexicon data** (`kosac/data/*.csv`) — Creative Commons Attribution-ShareAlike
  4.0 International (CC BY-SA 4.0), derived from KOSAC (Seoul National
  University). See [`src/kosac/data/LICENSE`](src/kosac/data/LICENSE).

If you use this lexicon in research, please cite KOSAC and this package.
