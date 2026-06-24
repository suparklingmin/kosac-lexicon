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
  return as_file(files('kosac.data').joinpath(filename))
