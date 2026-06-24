"""Citation and feature-glossary helpers.

The value summaries follow the KOSAC v1.0 annotation scheme; see the bundled
``kosac/data/readme.txt`` for the authoritative definitions.
"""

#: Short English summaries of each feature's label values.
FEATURE_VALUES = {
    'polarity': {
        'POS': 'positive sentiment',
        'NEG': 'negative sentiment',
        'NEUT': 'neutral sentiment',
        'COMP': 'complex / mixed polarity',
        'None': 'no polarity expressed',
    },
    'intensity': {
        'High': 'strong sentiment',
        'Medium': 'moderate sentiment',
        'Low': 'weak sentiment',
        'None': 'no intensity',
    },
    'expressive-type': {
        'dir-explicit': 'directly and explicitly expressed',
        'dir-speech': 'expressed through reported speech',
        'dir-action': 'expressed through a described action',
        'indirect': 'indirectly expressed',
        'writing-device': 'expressed via a rhetorical / writing device',
    },
    'nested-order': {
        '0': 'top-level (not nested) subjective expression',
        '1': 'nested one level deep',
        '2': 'nested two levels deep',
        '3': 'nested three levels deep',
    },
    'subjectivity-polarity': {
        'POS': 'positive subjectivity',
        'NEG': 'negative subjectivity',
        'NEUT': 'neutral subjectivity',
        'COMP': 'complex / mixed subjectivity',
    },
    'subjectivity-type': {
        'Judgment': 'evaluative judgment',
        'Emotion': 'emotion',
        'Argument': 'argument',
        'Intention': 'intention',
        'Agreement': 'agreement / disagreement',
        'Speculation': 'speculation',
        'Others': 'other subjectivity types',
    },
}


def describe_feature(feature):
  """Return ``{feature, values, reference}`` describing a feature's labels."""
  try:
    values = FEATURE_VALUES[feature]
  except KeyError:
    raise ValueError(
        f'unknown feature {feature!r}; choose from {sorted(FEATURE_VALUES)}'
    ) from None
  return {
      'feature': feature,
      'values': dict(values),
      'reference': 'KOSAC v1.0 annotation scheme (see kosac/data/readme.txt)',
  }


def citation(style='bibtex'):
  """Return a citation string for the KOSAC lexicon package."""
  if style != 'bibtex':
    raise ValueError("only 'bibtex' is supported")
  from . import __data_version__, __version__
  return (
      '@misc{kosac_lexicon,\n'
      '  title        = {KOSAC: a morpheme-level Korean sentiment lexicon},\n'
      '  author       = {Park, Sumin},\n'
      f'  year         = {{{__data_version__}}},\n'
      f'  note         = {{Python package kosac-lexicon {__version__}; '
      'derived from the Korean Sentiment Analysis Corpus (KOSAC)}},\n'
      '  howpublished = {\\url{https://github.com/suparklingmin/kosac-lexicon}}\n'
      '}'
  )
