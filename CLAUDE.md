# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`kosac-lexicon` — a pip-installable Python package distributing a morpheme-level **Korean sentiment lexicon** derived from the KOSAC (Korean Sentiment Analysis Corpus). The lexicon (frozen 2016 data) maps Korean morpheme N-grams (`surface/POS`, Sejong tagset) to distributions over six independent semantic features. The six CSVs ship as package data; a zero-config loader exposes them without file paths or Java. See `README.md` (English) and `src/kosac/data/readme.txt` / `README.ko.md` (Korean) for the linguistic derivation.

Distribution name is `kosac-lexicon`; **import name is `kosac`**.

## Commands

Everything runs from a venv with the package installed editable:

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"   # core + pytest/build/twine
.venv/bin/pytest                      # full suite (fast, no Java required)
.venv/bin/pytest tests/test_matching.py::test_wildcard_entry_is_escaped   # single test
.venv/bin/python -m build             # build wheel + sdist into dist/
.venv/bin/python -m twine check dist/*
```

Optional extras: `pip install -e ".[kiwi]"` (Kiwi POS tokenizer via `kiwipiepy` — pure pip, no Java) and `".[transformers]"` (HuggingFace tokenizer). The `dev` extra already includes `kiwipiepy`, so the Kiwi integration test in `tests/test_tokenizer_kiwi.py` runs; it `importorskip`s when Kiwi is absent.

## Architecture

`src/` layout — the package lives at `src/kosac/`, so `import kosac` requires an install (`pip install -e .`); this guarantees tests exercise the installed wheel and catch missing package-data.

### Core model (`src/kosac/lexicon.py`)
- `SentimentLexicon` base class wraps a pandas DataFrame indexed by `entry` (space-joined morphemes). Six concrete subclasses (`PolarityLexicon`, `IntensityLexicon`, `ExpressiveTypeLexicon`, `NestedOrderLexicon`, `SubjectivityPolarityLexicon`, `SubjectivityTypeLexicon`) + `GenericLexicon`. **Subclasses differ only by two class attributes**: `labels` (must match the value-columns of the corresponding CSV) and `_feature` (the bundled-data key).
- The CSV stores **only absolute label counts** (integers); on load the loader derives `freq` (row sum), `max.value` (idxmax over `self.labels`), and `max.prop` (max/sum) — none are stored. `pd.read_csv` uses `keep_default_na=False` so the literal `None` label (a valid polarity/intensity value) stays a string. `set_lexicon(min_freq, threshold)` re-filters from `self.original_lexicon` (so it can loosen, not just tighten). Legacy aliases: `get`=`get_entry`, `match`=`match_patterns`.
- Matching: `get_pattern()` builds a regex from entries sorted longest-N-gram / highest-`max.prop` first (via `utils.sort`); entries are `re.escape`d (some contain a regex-special `*`, e.g. `가*/JKS`). `match_patterns()` matches against a tokenized sentence; `get_sent_probs()` smooths counts, sums log-probs, applies `softmax`.

### Loader API — the low-level entry point
- `kosac.load_lexicon('polarity', ngrams=[1], min_freq=5)` (top-level factory; `kosac.FEATURES` lists valid names), or `PolarityLexicon.load(...)`. Both read bundled data via `src/kosac/_resources.py` (`importlib.resources.files('kosac.data')` + `as_file`).
- The `__init__(filepath=..., ngrams=...)` constructor still works for explicit CSV paths; `.load()` is layered on top, so passing a path is unchanged.

### High-level API (`analyzer.py`) — the recommended entry point
- `kosac.SentimentAnalyzer(features, tokenizer=None, ngrams=..., negation=, intensifier=, align=)` bundles one/all feature lexicons + a tokenizer (default `KiwiTokenizer`). `analyze(text)` returns a JSON-able dict: per feature, `{label, prob, probs, matches}` where each match has `entry`, char `span`, `text`, `max_value`, `negated`, `weight`. `analyze_batch` / `analyze_frame` (pandas) for many texts.
- `count(text)` / `count_batch` / `count_frame` are the **frequency method** (the social-science use from the title of the legacy notebook): tally matched words by their `max.value`, returning `{label, counts, proportions, total, matches}`. Same matcher as `analyze`, different aggregation.
- `select_matches()` is a greedy leftmost-longest, non-overlapping token matcher (uses `tokenizer.tokenize_with_offsets`) — it replaces regex `match_patterns` for the analyzer and yields char offsets.
- Negation/intensifier are opt-in window heuristics over token positions (`DEFAULT_NEGATIONS`/`DEFAULT_INTENSIFIERS`); negation only flips features whose labels include both POS and NEG. Aggregation is a per-entry weighted log-prob sum + softmax (equivalent to `get_sent_probs` when composition is off).

### Tooling
- `cli.py` (+`__main__.py`): `kosac` console script (`[project.scripts]`) — `analyze`/`features`/`citation`; `analyze` reads stdin when no text arg, `--compact` for JSONL.
- `sklearn.py`: `KosacVectorizer` (BaseEstimator/TransformerMixin) → label-probability features; needs the `[sklearn]` extra (friendly ImportError otherwise). Not imported by `__init__`.
- `info.py`: `citation()` (BibTeX) and `describe_feature()`/`FEATURE_VALUES` (label glossaries), re-exported from `kosac`.

### Other modules
- `tokenizers.py`: `Tokenizer` base (whitespace) → `KiwiTokenizer` (Kiwi, the recommended POS tokenizer; emits `surface/POS`), plus `HuggingFaceTokenizer`. **Heavy deps (kiwipiepy/transformers) are imported lazily inside constructors** so `import kosac` works without them; absent extras raise a friendly `ImportError`. Keep it that way — do not add top-level `import kiwipiepy`/`transformers`. Kiwi's tagset is Sejong-based and aligns with the lexicon's tags, but some symbol/web tags differ. (`get_ngrams` uses the pure-Python `utils.ngrams`; no `nltk`.)
- `corpora.py`: `Corpus(filepath)` reads a headerless `text,label` CSV (used by `update_from_corpus`).
- `utils.py`: `add_one`/`smooth` (Laplace), `longer_first`/`sort` (match order), `softmax`.

### Data & layout
- `src/kosac/data/*.csv` — the six lexicons (16,362 entries each: 3,476 unigrams / 6,579 bigrams / 6,307 trigrams), bundled into the wheel. Columns: `ngram` (morphemes joined by a space — the same form as the `entry` index, so no separator translation on load/save) and one **absolute-count** column per feature value. `freq`, `max.value`, and `max.prop` are derived by the loader, not stored. The original 2016 release stored relative frequencies; absolute count = `round(relative × freq)`.
- `tests/` — pytest suite; `conftest.py` provides a tiny inline-CSV `mini_lexicon` fixture (no Java) used by matching/inference tests. Bundled-data tests assert fixed counts (e.g. polarity unigrams == 3476).
- `examples/` — `quickstart.py` (current API) + `README.md`. `docs/` — the user manual (`manual.md`, `manual.ko.md`). `data/example.csv` — a tiny demo corpus. None of these ship in the wheel (sdist includes only `src/kosac`, `tests`, and the top-level docs/licence files).
- Old/heavy material that is not needed by the package — pre-package demo notebooks, the `draft.py`/legacy prototypes, the 23 MB source KOSAC corpus, the Komoran user dictionary, old PDFs/HTML — lives in a **gitignored `.archive/`** directory: present locally and in git history, but out of the repo and the package.

## Conventions & gotchas
- Package source uses **2-space indentation** (inherited from the original); tests use 4-space. Match the file you're editing.
- **Dual licensing**: code is MIT (`LICENSE`); the lexicon data is CC BY-SA 4.0 (`src/kosac/data/LICENSE`), derived from KOSAC (SNU). Preserve both when redistributing, and keep `data/LICENSE` shipping inside the wheel.
- Version policy: `kosac.__version__` is the SemVer package/API version; the data vintage is the separate `kosac.__data_version__` ("2016").
- When adding a new feature/lexicon: add the CSV to `src/kosac/data/`, a subclass with `labels`+`_feature`, and an entry in both `_resources.FEATURE_FILES` and `__init__._REGISTRY`.
