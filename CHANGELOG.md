# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html) on the
package/API (the underlying lexicon data is fixed; its vintage is exposed as
`kosac.__data_version__`).

## [1.0.0] — 2026-06-24

First packaged release of the KOSAC morpheme-level Korean sentiment lexicon
(data vintage 2016).

### Added
- Installable package `kosac-lexicon` with the six sentiment-feature lexicons
  bundled as package data and loaded via `kosac.load_lexicon(...)` /
  `<Lexicon>.load()` — no file paths required.
- `pandas`/`numpy`-only core install; the Kiwi POS tokenizer (`kiwipiepy`, no
  Java required) and HuggingFace tokenizer are optional extras
  (`kosac-lexicon[kiwi]`, `[transformers]`). N-gram generation is pure-Python,
  so `nltk` is no longer a dependency.
- pytest test suite covering loading, filtering, matching, and inference.

### Fixed
- `get_match_info` / `get_sent_probs` called a non-existent `self.match`
  (renamed to `match_patterns`).
- `lexicon.py` used `numpy` only via a star-import leak; now imported explicitly.
- Regex pattern building now escapes entries, so wildcard morphemes such as
  `가*/JKS` no longer raise `re.error`.
- `HuggingFaceTokenizer` was constructed incorrectly (`self = tokenizer`
  no-op and bad multiple inheritance); now composes an `AutoTokenizer`.
- Heavy tagger dependencies are imported lazily, so `import kosac` succeeds
  without Java or `transformers` installed.
