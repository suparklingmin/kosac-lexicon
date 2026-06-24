import re

import numpy as np
import pandas as pd

from .utils import smooth, sort, softmax


class SentimentLexicon:
  labels = []      # overridden by concrete subclasses
  _feature = None  # bundled-data key (see kosac/data/), set by subclasses

  def __init__(self, filepath=None, ngrams=[1]):
    self.ngrams = ngrams
    self.min_freq = 0
    self.threshold = 0.0

    if filepath:
      # keep_default_na=False so the literal 'None' label (a valid polarity /
      # intensity value) is not parsed as a missing value.
      df = pd.read_csv(filepath, keep_default_na=False)
      df['entry'] = df['ngram'].str.replace(';', ' ')

      # relative frequency -> absolute frequency
      for label in self.labels:
        df[label] = (df[label] * df['freq']).apply(round)

      df = df.sort_values('max.prop', ascending=False)
      df['ngram'] = df['entry'].str.count(' ') + 1
      df = df[df['ngram'].isin(self.ngrams)]
      df = df.sort_values('ngram', ascending=False)
      df.sort_values('entry', inplace=True)
      df.set_index('entry', inplace=True)
    else:
      df = pd.DataFrame()
      df.index.name = 'entry'

    self.original_lexicon = df
    self.lexicon = self.original_lexicon.copy()

  @classmethod
  def load(cls, ngrams=[1], min_freq=0, threshold=0.0):
    """Load this feature's lexicon from the CSV bundled with the package."""
    if cls._feature is None:
      raise TypeError(
          f'{cls.__name__} has no bundled data; use a concrete subclass such as '
          'PolarityLexicon, or pass an explicit filepath to the constructor.'
      )
    from ._resources import resource_path
    with resource_path(cls._feature) as path:
      lexicon = cls(filepath=str(path), ngrams=ngrams)
    if min_freq or threshold:
      lexicon.set_lexicon(min_freq=min_freq, threshold=threshold)
    return lexicon

  def __repr__(self):
    name = type(self).__name__
    return f'{name}(ngrams={self.ngrams}, min_freq={self.min_freq}, threshold={self.threshold})'

  def __eq__(self, other):
    self.lexicon == other.lexicon

  def __ne__(self, other):
    self.lexicon != other.lexicon

  def __add__(self, other):
    raise NotImplementedError

  def get_original_lexicon(self):
    return self.original_lexicon

  def set_lexicon(self, min_freq=0, threshold=0.0):
    # Re-applicable filter: always start from the originally loaded lexicon so the
    # threshold can be loosened as well as tightened. Falls back to the current
    # lexicon for from-scratch builds (where original_lexicon is empty).
    # TODO: frequency 대신 tf-idf
    df = self.original_lexicon if len(self.original_lexicon) else self.lexicon
    self.lexicon = df[(df['freq'] >= min_freq) & (df['max.prop'] > threshold)]
    self.min_freq = min_freq
    self.threshold = threshold

  def get_lexicon(self):
    return self.lexicon

  def get_size(self):
    return len(self.lexicon)

  def get_labels(self):
    return self.labels

  def get_entry(self, morph):
    return self.lexicon.loc[morph]

  def verify(self, morph, verbose=True):
    counts = self.lexicon.loc[morph, self.labels].astype('int')
    self.lexicon.loc[morph, 'freq'] = counts.sum()
    self.lexicon.loc[morph, 'max.value'] = counts.idxmax()
    self.lexicon.loc[morph, 'max.prop'] = counts.max() / counts.sum()
    if verbose:
      print(self.lexicon.loc[morph])

  def initialize_entry(self, morph, **kwargs):
    row = pd.Series(dtype='object')
    row['ngram'] = morph.count(' ') + 1
    row['freq'] = 0
    for label in self.labels:
      row[label] = 0

    counts = row[self.labels]
    row['freq'] = counts.sum()
    row['max.value'] = counts.idxmax()
    row['max.prop'] = 0.
    if len(self.lexicon.columns) == 0:
      # First insert into an empty lexicon: establish the columns first.
      self.lexicon = pd.DataFrame(columns=row.index)
      self.lexicon.index.name = 'entry'
    self.lexicon.loc[morph] = row

  def add_token(self, morph, tag, verbose=True):
    if morph not in self.lexicon.index:
      self.initialize_entry(morph)

    self.lexicon.loc[morph, tag] += 1
    self.verify(morph, verbose)

  def update(self, examples):
    for (morph, tag) in examples:
      self.add_token(morph, tag, verbose=False)

  def update_from_corpus(self, corpus, tokenizer):
    self.lexicon = self.original_lexicon.copy()
    self.lexicon['ngram'] = None
    self.lexicon['freq'] = None
    self.lexicon[self.labels] = None
    corpus.df['entry'] = corpus.df['text'].astype('str').apply(lambda x: tokenizer.get_ngrams(x, self.ngrams))
    examples = [pair for (_, pair) in corpus.df[['entry', 'label']].explode('entry').iterrows()]
    self.update(examples)

  def export_user_dict(self, dict_path='user_dictionary.txt'):
    unigrams = self.lexicon[self.lexicon['ngram'] == 1].index.tolist()
    with open(dict_path, 'w') as f:
      f.writelines('\n'.join(['\t'.join(unigram.split('/')) for unigram in unigrams]))
    print('USER_DICT PATH:', dict_path)
    self.dict_path = dict_path

  def get_pattern(self, sorting=True):
    my_lexicon = self.lexicon.copy()
    if sorting:
      sorts = sort(my_lexicon)
    else:
      sorts = my_lexicon

    # Entries are matched literally: some contain regex-special characters
    # (e.g. the wildcard '*' in '가*/JKS'), so each entry must be escaped.
    return re.compile('|'.join(re.escape(entry) for entry in sorts.index))

  def match_patterns(self, sentence, tokenizer, sorting=True):
    pattern = self.get_pattern(sorting)
    tagged = tokenizer.get_tokens_str(sentence)
    matches = pattern.findall(tagged)
    return matches

  def get_match_info(self, sentence, tokenizer, sorting=True):
    matches = self.match_patterns(sentence, tokenizer, sorting)
    result = [(match, self.lexicon.loc[match, 'max.value'], self.lexicon.loc[match, 'max.prop']) for match in matches]
    return result

  def get_smoothed_lexicon(self):
    return self.lexicon.apply(smooth, labels=self.labels, axis=1)

  def get_sent_probs(self, sentence, tokenizer, smoothing=True):
    matches = self.match_patterns(sentence, tokenizer)
    frequencies = self.lexicon.loc[matches].copy()
    if smoothing:
      smoothed = frequencies.apply(smooth, labels=self.labels, axis=1)
    else:
      smoothed = frequencies[self.labels]

    return softmax(np.log(smoothed).sum()).sort_values(ascending=False)

  # Legacy aliases for the original SentLex notebook API.
  get = get_entry
  match = match_patterns


class PolarityLexicon(SentimentLexicon):
  """Sentiment polarity: POS / NEG / NEUT / COMP (mixed) / None."""
  labels = ['COMP', 'NEG', 'NEUT', 'None', 'POS']
  _feature = 'polarity'

class IntensityLexicon(SentimentLexicon):
  """Sentiment intensity: High / Medium / Low / None."""
  labels = ['High', 'Low', 'Medium', 'None']
  _feature = 'intensity'

class ExpressiveTypeLexicon(SentimentLexicon):
  """How the sentiment is expressed (direct/indirect/writing-device, ...)."""
  labels = ['dir-action', 'dir-explicit', 'dir-speech', 'indirect', 'writing-device']
  _feature = 'expressive-type'

class NestedOrderLexicon(SentimentLexicon):
  """Nesting depth of the subjective expression (0–3)."""
  labels = ['0', '1', '2', '3']
  _feature = 'nested-order'

class SubjectivityPolarityLexicon(SentimentLexicon):
  """Polarity of the subjectivity: POS / NEG / NEUT / COMP."""
  labels = ['COMP', 'NEG', 'NEUT', 'POS']
  _feature = 'subjectivity-polarity'

class SubjectivityTypeLexicon(SentimentLexicon):
  """Type of subjectivity: Judgment / Emotion / Argument / Intention / ..."""
  labels = ['Agreement', 'Argument', 'Emotion', 'Intention', 'Judgment', 'Others', 'Speculation']
  _feature = 'subjectivity-type'

class GenericLexicon(SentimentLexicon):
  """A lexicon with user-defined labels (see :meth:`set_labels`)."""

  def set_labels(self, labels: list):
    """Set the label set for this lexicon."""
    self.labels = labels
