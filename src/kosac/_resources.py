"""Access to the lexicon CSV files bundled as package data (``kosac/data``)."""
from importlib.resources import as_file, files

FEATURE_FILES = {
    'polarity': 'polarity.csv',
    'intensity': 'intensity.csv',
    'expressive-type': 'expressive-type.csv',
    'nested-order': 'nested-order.csv',
    'subjectivity-polarity': 'subjectivity-polarity.csv',
    'subjectivity-type': 'subjectivity-type.csv',
}


def resource_path(feature):
  """Return a context manager yielding a real filesystem path to a bundled CSV.

  Works whether the package is installed as a directory or inside a zip, because
  ``as_file`` materialises a temporary file when needed (so ``pandas.read_csv``
  always receives a usable path).
  """
  try:
    filename = FEATURE_FILES[feature]
  except KeyError:
    raise ValueError(
        f'unknown feature {feature!r}; choose from {sorted(FEATURE_FILES)}'
    ) from None
  # Resolve from the ``kosac`` package (which has an ``__init__``) and descend
  # into ``data``, rather than ``files('kosac.data')``: ``kosac/data`` ships no
  # ``__init__``, so on Python 3.9 it is a namespace package whose ``spec.origin``
  # is ``None`` and ``importlib.resources.files`` raises (TypeError) on it.
  return as_file(files('kosac').joinpath('data', filename))
