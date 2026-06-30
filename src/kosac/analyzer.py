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

# Markers used by the optional negation/intensifier composition. Keyed by the
# tokenizer's ``surface/POS`` form; override via the analyzer constructor.
DEFAULT_NEGATIONS = frozenset({
    '안/MAG', '못/MAG', '않/VX', '못하/VX', '말/VX', '아니/VCN', '아니/VA',
})
DEFAULT_INTENSIFIERS = frozenset({
    '정말/MAG', '너무/MAG', '매우/MAG', '아주/MAG', '굉장히/MAG', '무척/MAG',
    '진짜/MAG', '엄청/MAG', '되게/MAG', '참/MAG',
})


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


def select_matches_multiscale(tokens, entry_set, ngram_lengths):
  """All (overlapping) n-gram windows present in ``entry_set``, every length.

  Unlike :func:`select_matches` (greedy leftmost-longest, non-overlapping), this
  yields *every* matching window, so a token contributes to its unigram and to any
  longer n-grams it is part of. Summing these multi-scale matches (instead of
  letting a long match suppress its unigrams) scores markedly better on held-out
  sentiment classification, which is why it is the analyzer's default. Same return
  shape as :func:`select_matches`.
  """
  lengths = sorted({n for n in ngram_lengths if n > 0})
  n_tokens = len(tokens)
  matches = []
  for length in lengths:
    for i in range(n_tokens - length + 1):
      entry = (tokens[i][0] if length == 1
               else ' '.join(tokens[k][0] for k in range(i, i + length)))
      if entry in entry_set:
        matches.append((entry, i, i + length - 1, tokens[i][1], tokens[i + length - 1][2]))
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
  scoring : {'multiscale', 'greedy'}
      ``'multiscale'`` (default) sums every overlapping n-gram match, which scores
      best on held-out sentiment classification; ``'greedy'`` is the legacy
      leftmost-longest non-overlapping matcher. Affects ``analyze``/``polarity_score``
      (``count`` always uses the non-overlapping matcher for word tallies).
  ngram_weights : dict[int, float], optional
      Per-length weights for multiscale matches (default 1.0 each).
  """

  def __init__(self, features='polarity', tokenizer=None, ngrams=DEFAULT_NGRAMS,
               min_freq=0, threshold=0.0, smoothing=True, align=False,
               negation=False, intensifier=False, window=2, intensifier_factor=2.0,
               negations=None, intensifiers=None, scoring='multiscale',
               ngram_weights=None):
    from . import load_lexicon, FEATURES

    if features == 'all':
      features = list(FEATURES)
    elif isinstance(features, str):
      features = [features]
    self.features = list(features)
    self.ngrams = list(ngrams)
    self.smoothing = smoothing
    self.negation = negation
    self.intensifier = intensifier
    self.window = window
    self.intensifier_factor = intensifier_factor
    if scoring not in ('multiscale', 'greedy'):
      raise ValueError("scoring must be 'multiscale' or 'greedy'")
    # 'multiscale' sums every overlapping n-gram (default, best classification);
    # 'greedy' keeps the legacy leftmost-longest non-overlapping matcher.
    self.scoring = scoring
    # Optional per-length weights for multiscale matches (default 1.0 each).
    self.ngram_weights = dict(ngram_weights or {})
    self.negations = DEFAULT_NEGATIONS if negations is None else frozenset(negations)
    self.intensifiers = DEFAULT_INTENSIFIERS if intensifiers is None else frozenset(intensifiers)
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

  def _polarity_feature(self):
    """Name of a loaded feature whose labels include both POS and NEG."""
    if 'polarity' in self.lexicons and \
        {'POS', 'NEG'} <= set(self.lexicons['polarity'].get_labels()):
      return 'polarity'
    for name, lexicon in self.lexicons.items():
      if {'POS', 'NEG'} <= set(lexicon.get_labels()):
        return name
    raise ValueError('no loaded feature has both POS and NEG labels; '
                     'predict_polarity/polarity_score need a polarity-type feature')

  def polarity_score(self, text):
    """Continuous polarity in [-1, 1]: ``P(POS) - P(NEG)`` (higher = more POS).

    The benchmark's decision axis — threshold it to get a binary label (see
    :meth:`predict_polarity`). Uses the multiscale scorer by default.
    """
    feature = self._polarity_feature()
    tokens = self.tokenizer.tokenize_with_offsets(str(text))
    probs = self._score(text, tokens, self.lexicons[feature])['probs']
    return probs.get('POS', 0.0) - probs.get('NEG', 0.0)

  def predict_polarity(self, text, threshold=0.0):
    """Binary polarity label: ``'POS'`` if :meth:`polarity_score` > ``threshold``."""
    return 'POS' if self.polarity_score(text) > threshold else 'NEG'

  def predict_polarity_batch(self, texts, threshold=0.0):
    """:meth:`predict_polarity` over an iterable of texts."""
    return [self.predict_polarity(text, threshold) for text in texts]

  def count(self, text):
    """Frequency-based analysis (the method common in social-science studies).

    Counts matched morphemes by their dominant label (``max.value``) and reports
    counts, proportions, and the most frequent label per feature.
    """
    text = str(text)
    tokens = self.tokenizer.tokenize_with_offsets(text)
    return {
        'text': text,
        'tokens': [token for token, _start, _end in tokens],
        'features': {feature: self._count(text, tokens, lexicon)
                     for feature, lexicon in self.lexicons.items()},
    }

  def count_batch(self, texts):
    """Frequency-count an iterable of strings. Returns a list of result dicts."""
    return [self.count(text) for text in texts]

  def count_frame(self, texts):
    """Frequency-count into a ``pandas.DataFrame``: ``<feature>.label``,
    ``<feature>.total``, and one ``<feature>.<label>`` count column per label."""
    import pandas as pd

    records = []
    for result in self.count_batch(texts):
      row = {'text': result['text']}
      for feature, counted in result['features'].items():
        row[f'{feature}.label'] = counted['label']
        row[f'{feature}.total'] = counted['total']
        for label, count in counted['counts'].items():
          row[f'{feature}.{label}'] = count
      records.append(row)
    return pd.DataFrame.from_records(records)

  def _count(self, text, tokens, lexicon):
    entry_set = set(lexicon.lexicon.index)
    selected = select_matches(tokens, entry_set, self.ngrams)
    labels = lexicon.get_labels()

    counts = {label: 0 for label in labels}
    matches = []
    for (entry, _ti, _tj, char_start, char_end) in selected:
      max_value = lexicon.lexicon.loc[entry, 'max.value']
      if max_value in counts:
        counts[max_value] += 1
      matches.append({
          'entry': entry,
          'span': [char_start, char_end],
          'text': text[char_start:char_end],
          'max_value': max_value,
          'max_prop': float(lexicon.lexicon.loc[entry, 'max.prop']),
      })

    total = sum(counts.values())
    proportions = {label: (counts[label] / total if total else 0.0) for label in labels}
    label = max(counts, key=counts.get) if total else None
    return {
        'label': label,
        'counts': counts,
        'proportions': proportions,
        'total': total,
        'matches': matches,
    }

  def _score(self, text, tokens, lexicon):
    entry_set = set(lexicon.lexicon.index)
    matcher = (select_matches_multiscale if self.scoring == 'multiscale'
               else select_matches)
    selected = matcher(tokens, entry_set, self.ngrams)
    labels = lexicon.get_labels()
    token_strs = [token for token, _start, _end in tokens]
    can_negate = self.negation and 'POS' in labels and 'NEG' in labels

    matches, weighted = [], []
    for (entry, ti, tj, char_start, char_end) in selected:
      negate = False
      weight = self.ngram_weights.get(tj - ti + 1, 1.0)
      if can_negate or self.intensifier:
        ctx = self._context_tokens(token_strs, ti, tj)
        if can_negate and any(t in self.negations for t in ctx):
          negate = True
        if self.intensifier and any(t in self.intensifiers for t in ctx):
          weight *= self.intensifier_factor
      matches.append({
          'entry': entry,
          'span': [char_start, char_end],
          'text': text[char_start:char_end],
          'max_value': lexicon.lexicon.loc[entry, 'max.value'],
          'max_prop': float(lexicon.lexicon.loc[entry, 'max.prop']),
          'negated': negate,
          'weight': weight,
      })
      weighted.append((entry, weight, negate))

    probs = self._aggregate(lexicon, weighted, labels)
    label = max(probs, key=probs.get) if probs else None
    return {
        'label': label,
        'prob': probs.get(label) if label is not None else None,
        'probs': probs,
        'matches': matches,
    }

  def _context_tokens(self, token_strs, ti, tj):
    """Tokens within ``self.window`` of a match's span, excluding the match."""
    lo = max(0, ti - self.window)
    hi = min(len(token_strs), tj + 1 + self.window)
    return token_strs[lo:ti] + token_strs[tj + 1:hi]

  def _aggregate(self, lexicon, weighted, labels):
    if not weighted:
      return {}
    pos_i = labels.index('POS') if 'POS' in labels else None
    neg_i = labels.index('NEG') if 'NEG' in labels else None
    accum = np.zeros(len(labels))
    for entry, weight, negate in weighted:
      row = lexicon.lexicon.loc[entry]
      if self.smoothing:
        dist = smooth(row, labels=labels).to_numpy(dtype=float)
      else:
        dist = row[labels].to_numpy(dtype=float)
      log_dist = np.log(dist)
      if negate and pos_i is not None and neg_i is not None:
        log_dist[pos_i], log_dist[neg_i] = log_dist[neg_i], log_dist[pos_i]
      accum += weight * log_dist
    probs = softmax(accum)
    return {label: float(prob) for label, prob in zip(labels, probs)}
