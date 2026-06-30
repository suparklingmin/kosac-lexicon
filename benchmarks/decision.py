"""Unified decision rules for binary (POS vs NEG) polarity scoring.

Every rule reduces a document to a single continuous score (higher = more POS)
plus a match count, so the benchmark never argmaxes over the 5 KOSAC labels for a
binary task. A separate threshold (see :mod:`benchmarks.metrics`) turns the score
into a label, which is what lets one knob absorb the NIKL class imbalance.

Rules
-----
logodds         Sum of ``log(POS+1) - log(NEG+1)`` over matched entries. Matches
                the reference harness; ignores NEUT/None/COMP and any composition.
softmax_margin  ``P(POS) - P(NEG)`` from the analyzer's smoothed log-prob softmax
                over ALL labels, so negation / intensifier / smoothing compose.
count_diff      ``#POS-dominant matches - #NEG-dominant matches`` (the cheap
                ``_count`` path collapsed to a binary axis).

A token list is ``[(surface/POS, char_start, char_end), ...]`` as produced by
``tokenizer.tokenize_with_offsets``; only the surface string is used by logodds /
count_diff, the offsets feed the analyzer's match spans for softmax_margin.
"""
import numpy as np

from kosac.analyzer import SentimentAnalyzer, select_matches
from kosac.utils import softmax

RULES = ('logodds', 'count_diff', 'softmax_margin')


def tokenize_docs(tokenizer, texts):
  """Tokenize many texts into ``(surface/POS, start, end)`` triple lists."""
  return [tokenizer.tokenize_with_offsets(str(text)) for text in texts]


def _entry_set(lexicon):
  return set(lexicon.lexicon.index)


def score_logodds(lexicon, token_lists, ngrams):
  """POS-vs-NEG log-odds per doc. Returns ``(scores, match_counts)``."""
  entries = _entry_set(lexicon)
  df = lexicon.lexicon
  delta = dict(zip(df.index,
                   np.log(df['POS'].astype(int) + 1) - np.log(df['NEG'].astype(int) + 1)))
  scores = np.zeros(len(token_lists))
  counts = np.zeros(len(token_lists), dtype=int)
  for i, toks in enumerate(token_lists):
    matches = select_matches(toks, entries, ngrams)
    counts[i] = len(matches)
    scores[i] = sum(delta[entry] for (entry, *_rest) in matches)
  return scores, counts


def score_logodds_multiscale(lexicon, token_lists, weights):
  """Weighted multi-scale log-odds: each n-gram length scored independently
  (sliding over ALL windows, overlapping across scales) then summed with
  per-length ``weights`` (a ``{length: weight}`` dict).

  Unlike :func:`score_logodds` (which rides the greedy leftmost-longest matcher,
  so a trigram suppresses its unigrams), here every unigram always fires and the
  higher orders are *added* with their own weight — letting unigrams' OOD
  robustness and higher orders' in-domain signal be dialed in together.
  Returns ``(scores, match_counts)``.
  """
  df = lexicon.lexicon
  delta = dict(zip(df.index,
                   np.log(df['POS'].astype(int) + 1) - np.log(df['NEG'].astype(int) + 1)))
  entries = set(df.index)
  lengths = [L for L in sorted(weights) if weights[L] != 0]
  scores = np.zeros(len(token_lists))
  counts = np.zeros(len(token_lists), dtype=int)
  for i, toks in enumerate(token_lists):
    forms = [t[0] for t in toks]
    s, c = 0.0, 0
    for L in lengths:
      w = weights[L]
      for j in range(len(forms) - L + 1):
        entry = forms[j] if L == 1 else ' '.join(forms[j:j + L])
        if entry in entries:
          s += w * delta[entry]
          c += 1
    scores[i] = s
    counts[i] = c
  return scores, counts


def score_count_diff(lexicon, token_lists, ngrams):
  """#POS-dominant minus #NEG-dominant matches. Returns ``(scores, counts)``."""
  entries = _entry_set(lexicon)
  max_value = lexicon.lexicon['max.value'].to_dict()
  scores = np.zeros(len(token_lists))
  counts = np.zeros(len(token_lists), dtype=int)
  for i, toks in enumerate(token_lists):
    matches = select_matches(toks, entries, ngrams)
    counts[i] = len(matches)
    s = 0
    for (entry, *_rest) in matches:
      mv = max_value[entry]
      if mv == 'POS':
        s += 1
      elif mv == 'NEG':
        s -= 1
    scores[i] = s
  return scores, counts


def make_analyzer(tokenizer, ngrams, negation=False, intensifier=False,
                  smoothing=True, window=2, intensifier_factor=2.0):
  """A :class:`SentimentAnalyzer` whose ``_score`` we reuse with custom lexicons.

  Its bundled polarity lexicon is never used here — ``_score(text, tokens, lex)``
  takes the lexicon as an argument, so any lexicon can be plugged in per call.
  """
  return SentimentAnalyzer('polarity', tokenizer=tokenizer, ngrams=list(ngrams),
                           smoothing=smoothing, negation=negation,
                           intensifier=intensifier, window=window,
                           intensifier_factor=intensifier_factor)


def score_softmax_margin(analyzer, lexicon, token_lists, texts=None):
  """``P(POS) - P(NEG)`` from the analyzer's smoothed log-prob softmax.

  Mirrors :meth:`SentimentAnalyzer._aggregate` (smoothing, negation POS/NEG swap,
  intensifier weighting) but precomputes each entry's log-distribution once as a
  NumPy array, avoiding the per-match ``DataFrame.loc`` lookups that make the
  analyzer's own path ~100x slower over 50k documents. Returns ``(scores, counts)``.
  """
  labels = lexicon.get_labels()
  pos_i = labels.index('POS') if 'POS' in labels else None
  neg_i = labels.index('NEG') if 'NEG' in labels else None
  ngrams = analyzer.ngrams
  entries = _entry_set(lexicon)

  count_matrix = lexicon.lexicon[labels].to_numpy(dtype=float)
  if analyzer.smoothing:
    smoothed = (count_matrix + 1) / (count_matrix + 1).sum(axis=1, keepdims=True)
  else:
    smoothed = count_matrix
  with np.errstate(divide='ignore'):
    log_dists = np.log(smoothed)
  log_of = {entry: log_dists[i] for i, entry in enumerate(lexicon.lexicon.index)}

  can_negate = analyzer.negation and pos_i is not None and neg_i is not None
  use_ctx = can_negate or analyzer.intensifier
  negations, intensifiers = analyzer.negations, analyzer.intensifiers
  window, factor = analyzer.window, analyzer.intensifier_factor

  scores = np.zeros(len(token_lists))
  counts = np.zeros(len(token_lists), dtype=int)
  for d, toks in enumerate(token_lists):
    matches = select_matches(toks, entries, ngrams)
    counts[d] = len(matches)
    if not matches:
      continue
    token_strs = [t[0] for t in toks]
    accum = np.zeros(len(labels))
    for (entry, ti, tj, *_rest) in matches:
      log_dist = log_of[entry].copy()
      weight = 1.0
      if use_ctx:
        lo, hi = max(0, ti - window), min(len(token_strs), tj + 1 + window)
        ctx = token_strs[lo:ti] + token_strs[tj + 1:hi]
        if can_negate and any(t in negations for t in ctx):
          log_dist[pos_i], log_dist[neg_i] = log_dist[neg_i], log_dist[pos_i]
        if analyzer.intensifier and any(t in intensifiers for t in ctx):
          weight = factor
      accum += weight * log_dist
    probs = softmax(accum)
    scores[d] = (probs[pos_i] if pos_i is not None else 0.0) - \
                (probs[neg_i] if neg_i is not None else 0.0)
  return scores, counts


def score_ensemble(per_lexicon, weights=None):
  """Combine several ``(scores, counts)`` results into one weighted score.

  ``per_lexicon`` is a list of ``(scores, counts)`` tuples (e.g. one logodds
  result per lexicon). Scores are summed with ``weights`` (default equal);
  match counts are summed so coverage reflects any contributing lexicon.
  """
  if not per_lexicon:
    raise ValueError('score_ensemble needs at least one (scores, counts) result')
  if weights is None:
    weights = [1.0] * len(per_lexicon)
  scores = sum(w * np.asarray(s) for w, (s, _c) in zip(weights, per_lexicon))
  counts = sum(np.asarray(c) for (_s, c) in per_lexicon)
  return scores, counts
