# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html) on the
package/API (the underlying lexicon data is fixed; its vintage is exposed as
`kosac.__data_version__`).

## [0.5.0] — 2026-06-30

Sixth pre-release (beta). A performance release for **polarity classification**,
driven by the benchmark campaign in `benchmarks/` (NSMC in-domain + NIKL
out-of-domain). The frozen 2016 KOSAC data is unchanged; the gains come from a
better scorer and an optional derived lexicon.

### Added
- **Multi-scale scoring** (`SentimentAnalyzer(..., scoring='multiscale')`, now the
  **default**): sums every overlapping n-gram match instead of the greedy
  leftmost-longest non-overlapping matcher, so a long match no longer suppresses
  its unigrams. Strictly beats the old scorer on all four held-out metrics. Pass
  `scoring='greedy'` for the legacy behavior, or `ngram_weights={1:…,2:…,3:…}` to
  weight the scales. `analyze()`/`polarity_score()` use it; `count()` stays
  non-overlapping (correct for word tallies).
- **`polarity-blend`** lexicon (`load_lexicon('polarity-blend')` /
  `SentimentAnalyzer('polarity-blend')`): a derived POS/NEG lexicon — the frozen
  KOSAC seeds blended with an NSMC-learned lexicon — shipped gzipped
  (`data/polarity-blend.csv.gz`, ~0.9 MB). Far stronger than the frozen polarity
  lexicon out of domain (NIKL balanced-acc 0.53 → 0.73). It is loadable by name
  but is **not** one of the six canonical `FEATURES`, so `analyzer('all')` is
  unchanged. Regenerate (or build the larger "champion" variant) with
  `python -m benchmarks.build_shipped_blend [--full]`. Derived data is CC BY-SA
  (see `data/polarity-blend.NOTICE`).
- **Convenience API**: `SentimentAnalyzer.polarity_score(text)` (continuous
  `P(POS) − P(NEG)`), `predict_polarity(text, threshold=0.0)`, and
  `predict_polarity_batch(...)`.

### Changed
- Default analyzer scoring is now multi-scale (see above). For unigram-only use
  (`ngrams=[1]`) this is identical to before; it differs only when bigrams/
  trigrams are matched.

## [0.4.1] — 2026-06-25

Fifth pre-release (beta). Fixes bundled-data loading on Python 3.9 and makes
sentence scoring robust on out-of-vocabulary input.

### Fixed
- Loading bundled lexicons (`load_lexicon` / `Lexicon.load`) raised `TypeError`
  on **Python 3.9** (a supported version): `kosac/data` ships no `__init__`, so
  `importlib.resources.files('kosac.data')` hit a namespace package whose
  `spec.origin` is `None`. The data is now resolved from the `kosac` package
  (`files('kosac').joinpath('data', …)`), which loads on 3.9–3.12+.
- `SentimentLexicon.get_sent_probs()` now returns a uniform distribution for a
  sentence with no lexicon matches, instead of raising on the empty match frame.
  An out-of-vocabulary sentence carries no evidence, so every label is equally
  likely.

## [0.4.0] — 2026-06-25

Fourth pre-release (beta). Space-joined data CSVs; `save()` round-trips
punctuation morphemes.

### Changed
- Bundled data CSVs now join an N-gram's morphemes with a **space** (matching the
  in-memory `entry` index) instead of `;`, so the loader and `save()` no longer
  translate separators.

### Fixed
- `save()` now round-trips entries whose surface is itself `;` — e.g. the
  punctuation morpheme `;/SP` common in web text (and in corpus-built lexicons
  like the NSMC example). The old `;`-joined CSV format mangled them.

## [0.3.0] — 2026-06-25

Third pre-release (beta). Faster corpus-built lexicons and an NSMC worked example.

### Added
- `Tokenizer.tokenize_batch()` / `get_ngrams_batch()` tokenize many sentences at
  once; `KiwiTokenizer` overrides them to use Kiwi's parallel batch API (results
  are identical to per-sentence tokenizing, just faster).

### Changed
- `SentimentLexicon.update_from_corpus()` now tallies counts with a `Counter` and
  assembles the lexicon in one vectorized pass instead of a per-token pandas
  update, and tokenizes the corpus in one batch. Building from a large corpus is
  dramatically faster (a 150k-review corpus that took ~20 min now builds in ~15 s);
  results are identical. It also fails loud (`ValueError`) when the corpus carries
  a label the lexicon doesn't declare, and the rebuild now holds exactly the
  N-grams observed in the corpus (stale all-`None` rows from the previous index are
  no longer kept).

### Documentation
- New tutorial and runnable example (`examples/nsmc_lexicon.py`) building a POS/NEG
  lexicon from the NSMC movie-review corpus end to end.

## [0.2.0] — 2026-06-25

Second pre-release (beta).

### Added
- `SentimentLexicon.save(filepath)` writes a lexicon to the package's CSV format
  (`ngram` + one count column per label). The constructor reads it back —
  concrete subclasses use their declared labels, and `GenericLexicon` now infers
  labels from the CSV columns, so custom lexicons round-trip via CSV.
- `SentimentLexicon.update_from_corpus()` gains optional `pos_tag`, `min_freq`,
  and `max_value_threshold` arguments to filter a corpus-built lexicon. `pos_tag`
  keeps only N-grams containing at least one token of an allowed POS (e.g. content
  words), dropping pure function-morpheme entries (`이/MM`, `다/EF`) while keeping
  mixed constructions like `ㄹ/ETM 수/NNB 있/VV`. Backward compatible (no
  arguments → unchanged).

### Changed
- Bundled data CSVs (`kosac/data/*.csv`) now store **only absolute integer
  counts** (`ngram` + one column per label) instead of relative frequencies;
  `freq`, `max.value`, and `max.prop` are no longer stored — the loader derives
  them. This preserves tied top labels and makes the data mergeable/extensible.
  The package API is unchanged — loaded values are identical (absolute count =
  `round(relative × freq)` from the original 2016 release).

### Documentation
- Sphinx documentation site (guides + auto-generated API reference) deployed to
  GitHub Pages, with five tutorials (getting started, counting, machine learning,
  negation/intensifier, custom lexicon).

## [0.1.0] — 2026-06-24

First pre-release (beta) of the KOSAC morpheme-level Korean sentiment lexicon
(data vintage 2016), for review by colleagues and beta users ahead of the 1.0
release. The API may still change.

### Added
- Installable package `kosac-lexicon` with the six sentiment-feature lexicons
  bundled as package data and loaded via `kosac.load_lexicon(...)` /
  `<Lexicon>.load()` — no file paths required.
- `pandas`/`numpy`-only core install; the Kiwi POS tokenizer (`kiwipiepy`, no
  Java required) and HuggingFace tokenizer are optional extras
  (`kosac-lexicon[kiwi]`, `[transformers]`). N-gram generation is pure-Python,
  so `nltk` is no longer a dependency.
- `SentimentAnalyzer`: bundles lexicons + a tokenizer and scores text in one
  call, across one or all six features, returning the top label, full
  distribution, and matched entries with character spans; `analyze_batch` /
  `analyze_frame` (pandas) helpers.
- Opt-in negation and intensifier composition, and `align=` to seed Kiwi's user
  dictionary from the lexicon (`KiwiTokenizer.from_lexicon`).
- Frequency-count analysis (`SentimentAnalyzer.count` / `count_batch` /
  `count_frame`, and `kosac analyze --count`): counts of matched words per label
  with proportions — the method common in social-science studies.
- `kosac` command-line interface (`analyze` / `features` / `citation`), a
  scikit-learn `KosacVectorizer` (`[sklearn]` extra), and `kosac.citation()` /
  `kosac.describe_feature()` helpers.
- `set_lexicon` now re-filters from the originally loaded lexicon, so a
  threshold can be loosened as well as tightened; legacy `lex.get` / `lex.match`
  aliases for the original notebook API.
- pytest test suite covering loading, filtering, matching, inference, the
  analyzer, CLI, and scikit-learn.

### Fixed
- The `'None'` label (557 polarity / many intensity entries) was parsed as a
  missing value by `pandas.read_csv` and lost from `max.value`; loading now uses
  `keep_default_na=False`.
- `get_match_info` / `get_sent_probs` called a non-existent `self.match`
  (renamed to `match_patterns`).
- `lexicon.py` used `numpy` only via a star-import leak; now imported explicitly.
- Regex pattern building now escapes entries, so wildcard morphemes such as
  `가*/JKS` no longer raise `re.error`.
- `HuggingFaceTokenizer` was constructed incorrectly (`self = tokenizer`
  no-op and bad multiple inheritance); now composes an `AutoTokenizer`.
- `initialize_entry` could not insert into an empty lexicon (no columns);
  building a lexicon from scratch with `add_token` / `update` now works.
- Heavy tagger dependencies are imported lazily, so `import kosac` succeeds
  without Java or `transformers` installed.
