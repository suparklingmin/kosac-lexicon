# Examples

- [`quickstart.py`](quickstart.py) — a runnable end-to-end demo using the current
  `kosac` API (`kosac.load_lexicon(...)` and `SentimentAnalyzer`). Start here.
- [`nsmc_lexicon.py`](nsmc_lexicon.py) — build a **new** POS/NEG lexicon from the
  NSMC movie-review corpus (downloaded on first run), then inspect and score with
  it. Shows `update_from_corpus` on a real 150k-review corpus. Needs `[kiwi]`.
  Try `python examples/nsmc_lexicon.py --limit 20000` for a quick run.
- [`nikl_lexicon.py`](nikl_lexicon.py) — build a **3-class**
  POS/NEUT/NEG lexicon from a corpus of *scored expressions* (a data frame with
  `expression_form` + a −2..2 `expression_score`, e.g. a 모두의 말뭉치 / NIKL export).
  Shows the score→label mapping into `update_from_corpus`. Needs `[kiwi]`. Runs on
  a built-in sample with no args; pass `--data EXSA2002108040.csv` for your own.

For the full reference, see the manual: [English](../docs/manual.md) ·
[한국어](../docs/manual.ko.md).
