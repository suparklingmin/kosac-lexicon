# Examples

- [`quickstart.py`](quickstart.py) — a runnable end-to-end demo using the current
  `kosac` API (`kosac.load_lexicon(...)` and `SentimentAnalyzer`). Start here.
- [`nsmc_lexicon.py`](nsmc_lexicon.py) — build a **new** POS/NEG lexicon from the
  NSMC movie-review corpus (downloaded on first run), then inspect and score with
  it. Shows `update_from_corpus` on a real 150k-review corpus. Needs `[kiwi]`.
  Try `python examples/nsmc_lexicon.py --limit 20000` for a quick run.

For the full reference, see the manual: [English](../docs/manual.md) ·
[한국어](../docs/manual.ko.md).
