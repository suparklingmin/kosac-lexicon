import pandas as pd
import re

from .utils import *

class SentimentLexicon:
  def __init__(self, filepath=None, ngrams=[1]):
    self.ngrams = ngrams

    if filepath:
      df = pd.read_csv(filepath)
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
    df = self.lexicon
    # TODO: frequency 대신 tf-idf
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

  def export_user_dict(self, dict_path='./tagger/user_dictionary.txt'):
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
    
    return re.compile('|'.join(sorts.index))

  def match(self, sentence, tagger, sorting=True):
    pattern = self.get_pattern(sorting)
    tagged = tagger.tokenize(sentence)
    matches = pattern.findall(tagged)
    return matches

  def get_match_info(self, sentence, tagger, sorting=True):
    matches = self.match(sentence, tagger, sorting)
    result = [(match, self.lexicon.loc[match, 'max.value'], self.lexicon.loc[match, 'max.prop']) for match in matches]
    return result

  def get_smoothed_lexicon(self):
    return self.lexicon.apply(smooth, labels=self.labels, axis=1)

  def get_sent_probs(self, sentence, tagger, smoothing=True):
    matches = self.match(sentence, tagger)
    frequencies = self.lexicon.loc[matches].copy()
    if smoothing:
      smoothed = frequencies.apply(smooth, labels=self.labels, axis=1)
    else:
      smoothed = frequencies

    return softmax(np.log(smoothed).sum()).sort_values(ascending=False)

class PolarityLexicon(SentimentLexicon):
  labels = ['COMP', 'NEG', 'NEUT', 'None', 'POS'] 

class IntensityLexicon(SentimentLexicon):
  labels = ['High', 'Low', 'Medium', 'None']

class ExpressiveTypeLexicon(SentimentLexicon):
  labels = ['dir-action', 'dir-explicit', 'dir-speech', 'indirect', 'writing-device']

class NestedOrderLexicon(SentimentLexicon):
  labels = ['0', '1', '2', '3']

class SubjectivityPolarityLexicon(SentimentLexicon):
  labels = ['COMP', 'NEG', 'NEUT', 'POS']

class SubjectivityTypeLexicon(SentimentLexicon):
  labels = ['Agreement', 'Argument', 'Emotion', 'Intention', 'Judgment', 'Others', 'Speculation']

class GenericLexicon(SentimentLexicon):
  def set_labels(self, labels:list):
    self.labels = labels