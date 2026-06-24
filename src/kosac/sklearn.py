"""scikit-learn integration: use the KOSAC lexicon as a feature extractor.

Requires the ``sklearn`` extra (``pip install kosac-lexicon[sklearn]``)::

    from kosac.sklearn import KosacVectorizer
    from sklearn.pipeline import make_pipeline
    from sklearn.linear_model import LogisticRegression

    clf = make_pipeline(KosacVectorizer('all'), LogisticRegression())
    clf.fit(texts, labels)
"""
try:
  from sklearn.base import BaseEstimator, TransformerMixin
except ImportError as exc:  # pragma: no cover
  raise ImportError(
      'kosac.sklearn requires `pip install kosac-lexicon[sklearn]`.'
  ) from exc

import numpy as np


class KosacVectorizer(BaseEstimator, TransformerMixin):
  """Transform Korean text into KOSAC label-probability features.

  Each output column is one ``<feature>=<label>`` probability. Constructor
  arguments mirror :class:`kosac.SentimentAnalyzer`.
  """

  def __init__(self, features='polarity', ngrams=(1, 2, 3), min_freq=0,
               negation=False, intensifier=False, align=False, tokenizer=None):
    self.features = features
    self.ngrams = ngrams
    self.min_freq = min_freq
    self.negation = negation
    self.intensifier = intensifier
    self.align = align
    self.tokenizer = tokenizer

  def fit(self, X=None, y=None):
    from . import SentimentAnalyzer
    self.analyzer_ = SentimentAnalyzer(
        self.features, tokenizer=self.tokenizer, ngrams=self.ngrams,
        min_freq=self.min_freq, negation=self.negation,
        intensifier=self.intensifier, align=self.align)
    self.columns_ = [f'{feature}={label}'
                     for feature, lexicon in self.analyzer_.lexicons.items()
                     for label in lexicon.get_labels()]
    return self

  def transform(self, X):
    rows = []
    for result in self.analyzer_.analyze_batch(list(X)):
      scored = result['features']
      row = [scored[feature]['probs'].get(label, 0.0)
             for feature, lexicon in self.analyzer_.lexicons.items()
             for label in lexicon.get_labels()]
      rows.append(row)
    return np.asarray(rows, dtype=float)

  def get_feature_names_out(self, input_features=None):
    return np.asarray(self.columns_, dtype=object)
