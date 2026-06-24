"""High-level sentiment analysis built on top of the KOSAC lexicons.

:class:`SentimentAnalyzer` bundles one or more feature lexicons with a tokenizer
and scores raw text in a single call — including KOSAC's six semantic features at
once. It is the recommended entry point for end-to-end use::

    from kosac import SentimentAnalyzer
    analyzer = SentimentAnalyzer('all')            # needs a tokenizer (Kiwi by default)
    analyzer.analyze('이 영화 정말 좋다')
"""
import numpy as np

from .utils import smooth, softmax

# Default morpheme n-gram lengths to match (unigram .. trigram).
DEFAULT_NGRAMS = (1, 2, 3)


def select_matches(tokens, entry_set, ngram_lengths):
  """Greedy leftmost-longest, non-overlapping match over tokenized text.

  ``tokens`` is a list of ``(token_str, char_start, char_end)``. Returns a list
  of ``(entry, tok_start_idx, tok_end_idx, char_start, char_end)``.
  """
  lengths = sorted({n for n in ngram_lengths if n > 0}, reverse=True)
  n_tokens = len(tokens)
  matches = []
  i = 0
  while i < n_tokens:
    for length in lengths:
      if i + length > n_tokens:
        continue
      entry = ' '.join(tokens[k][0] for k in range(i, i + length))
      if entry in entry_set:
        matches.append((entry, i, i + length - 1, tokens[i][1], tokens[i + length - 1][2]))
        i += length
        break
    else:
      i += 1
  return matches


class SentimentAnalyzer:
  """Score Korean text against one or more KOSAC feature lexicons.

  Parameters
  ----------
  features : str | iterable[str]
      A feature name, an iterable of names, or ``'all'`` for every feature in
      :data:`kosac.FEATURES`.
  tokenizer : Tokenizer, optional
      Defaults to :class:`kosac.tokenizers.KiwiTokenizer` (requires the ``kiwi``
      extra). Any object with ``tokenize_with_offsets`` works.
  ngrams, min_freq, threshold :
      Forwarded to the underlying lexicons.
  smoothing : bool
      Apply add-one smoothing before aggregating (matches ``get_sent_probs``).
  """

  def __init__(self, features='polarity', tokenizer=None, ngrams=DEFAULT_NGRAMS,
               min_freq=0, threshold=0.0, smoothing=True, align=False):
    from . import load_lexicon, FEATURES

    if features == 'all':
      features = list(FEATURES)
    elif isinstance(features, str):
      features = [features]
    self.features = list(features)
    self.ngrams = list(ngrams)
    self.smoothing = smoothing
    self.lexicons = {
        feature: load_lexicon(feature, ngrams=self.ngrams, min_freq=min_freq, threshold=threshold)
        for feature in self.features
    }

    if tokenizer is None:
      from .tokenizers import KiwiTokenizer
      if align:
        # Seed Kiwi's user dictionary with the lexicon's unigrams so segmentation
        # tends to match the lexicon. All features share the same entries, so any
        # one lexicon works as the seed.
        seed = next(iter(self.lexicons.values()))
        tokenizer = KiwiTokenizer.from_lexicon(seed)
      else:
        tokenizer = KiwiTokenizer()
    self.tokenizer = tokenizer

  def analyze(self, text):
    """Analyze a single string. Returns a JSON-serialisable dict."""
    text = str(text)
    tokens = self.tokenizer.tokenize_with_offsets(text)
    return {
        'text': text,
        'tokens': [token for token, _start, _end in tokens],
        'features': {feature: self._score(text, tokens, lexicon)
                     for feature, lexicon in self.lexicons.items()},
    }

  def analyze_batch(self, texts):
    """Analyze an iterable of strings. Returns a list of result dicts."""
    return [self.analyze(text) for text in texts]

  def analyze_frame(self, texts):
    """Analyze an iterable of strings into a tidy ``pandas.DataFrame``.

    One row per text; ``<feature>.label`` / ``<feature>.prob`` columns.
    """
    import pandas as pd

    records = []
    for result in self.analyze_batch(texts):
      row = {'text': result['text']}
      for feature, scored in result['features'].items():
        row[f'{feature}.label'] = scored['label']
        row[f'{feature}.prob'] = scored['prob']
      records.append(row)
    return pd.DataFrame.from_records(records)

  def _score(self, text, tokens, lexicon):
    entry_set = set(lexicon.lexicon.index)
    selected = select_matches(tokens, entry_set, self.ngrams)
    labels = lexicon.get_labels()

    matches = [{
        'entry': entry,
        'span': [char_start, char_end],
        'text': text[char_start:char_end],
        'max_value': lexicon.lexicon.loc[entry, 'max.value'],
        'max_prop': float(lexicon.lexicon.loc[entry, 'max.prop']),
    } for (entry, _ti, _tj, char_start, char_end) in selected]

    probs = self._aggregate(lexicon, [m['entry'] for m in matches], labels)
    label = max(probs, key=probs.get) if probs else None
    return {
        'label': label,
        'prob': probs.get(label) if label is not None else None,
        'probs': probs,
        'matches': matches,
    }

  def _aggregate(self, lexicon, entries, labels):
    if not entries:
      return {}
    rows = lexicon.lexicon.loc[entries]
    if self.smoothing:
      distribution = rows.apply(smooth, labels=labels, axis=1)
    else:
      distribution = rows[labels]
    probs = softmax(np.log(distribution).sum())
    return {label: float(probs[label]) for label in labels}
