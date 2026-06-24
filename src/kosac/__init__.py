"""KOSAC — a morpheme-level Korean sentiment lexicon.

Quickstart::

    import kosac

    lex = kosac.load_lexicon('polarity', ngrams=[1], min_freq=5)
    lex.get_entry('힘/NNG')

The six sentiment features are available via :data:`kosac.FEATURES`.
"""
from ._resources import FEATURE_FILES
from .corpora import Corpus
from .lexicon import (
    ExpressiveTypeLexicon,
    GenericLexicon,
    IntensityLexicon,
    NestedOrderLexicon,
    PolarityLexicon,
    SentimentLexicon,
    SubjectivityPolarityLexicon,
    SubjectivityTypeLexicon,
)

__version__ = '0.1.0'
__data_version__ = '2016'

_REGISTRY = {
    'polarity': PolarityLexicon,
    'intensity': IntensityLexicon,
    'expressive-type': ExpressiveTypeLexicon,
    'nested-order': NestedOrderLexicon,
    'subjectivity-polarity': SubjectivityPolarityLexicon,
    'subjectivity-type': SubjectivityTypeLexicon,
}

#: The sentiment features that ship with the package.
FEATURES = tuple(_REGISTRY)


def load_lexicon(feature, ngrams=[1], min_freq=0, threshold=0.0):
  """Load a bundled KOSAC lexicon by feature name.

  Parameters
  ----------
  feature : str
      One of :data:`kosac.FEATURES` (e.g. ``'polarity'``, ``'intensity'``).
  ngrams : list[int]
      Which n-gram lengths to keep (default: unigrams only).
  min_freq, threshold : optional filtering forwarded to ``set_lexicon``.
  """
  try:
    cls = _REGISTRY[feature]
  except KeyError:
    raise ValueError(
        f'unknown feature {feature!r}; choose from {sorted(_REGISTRY)}'
    ) from None
  return cls.load(ngrams=ngrams, min_freq=min_freq, threshold=threshold)


from .analyzer import SentimentAnalyzer  # noqa: E402  (after load_lexicon/FEATURES)
from .info import FEATURE_VALUES, citation, describe_feature  # noqa: E402

__all__ = [
    'SentimentAnalyzer',
    'citation',
    'describe_feature',
    'FEATURE_VALUES',
    'SentimentLexicon',
    'PolarityLexicon',
    'IntensityLexicon',
    'ExpressiveTypeLexicon',
    'NestedOrderLexicon',
    'SubjectivityPolarityLexicon',
    'SubjectivityTypeLexicon',
    'GenericLexicon',
    'Corpus',
    'load_lexicon',
    'FEATURES',
    'FEATURE_FILES',
    '__version__',
    '__data_version__',
]
