import pandas as pd
import numpy as np
from konlpy.tag import Okt, Mecab, Kkma, Komoran
import re
from nltk import ngrams

# Smoothing
def add_one(x:np.array):
  return (x+1)/(x+1).sum()

# TODO: SentLex method로 바꾸기 (self.labels 사용하기)
def smooth(row:pd.Series, func=add_one):
  counts = row[['COMP', 'NEG', 'NEUT', 'None', 'POS']].astype('int')
  return func(counts)

# Sorting
def longer_first(df:pd.DataFrame):
  df = df.sort_values('max.prop', ascending=False)
  df = df.sort_values('ngram', ascending=False)
  return df

def sort(df:pd.DataFrame, func=longer_first):
  return func(df)

# Inference
def softmax(x:np.array):
  x = x - x.max()
  return np.exp(x) / sum(np.exp(x))

class Corpus:
  # TODO: HuggingFace Datasets 라이브러리 추가하기
  def __init__(self, filepath='./data/example.csv'):
    self.df = pd.read_csv(filepath, names=('text', 'label'))
    self.labels = self.df['label'].unique().tolist()
  
  def get_label(self):
    return self.labels

class SentLex:
  def __init__(self, filepath=None, ngrams=[1,2,3]):
    self.ngrams = ngrams
    # TODO: subclass로 PolarityLex 만들기
    self.labels = ['COMP', 'NEG', 'NEUT', 'None', 'POS'] 

    if filepath:
      df = pd.read_csv(filepath)
      df['entry'] = df['ngram'].str.replace(';', ' ')
      df['COMP'] = (df['COMP'] * df['freq']).astype('int')
      df['NEG'] = (df['NEG'] * df['freq']).astype('int')
      df['NEUT'] = (df['NEUT'] * df['freq']).astype('int')
      df['None'] = (df['None'] * df['freq']).astype('int')
      df['POS'] = (df['POS'] * df['freq']).astype('int')
      df = df[~df['ngram'].str.contains('*/', regex=False)]
      # df = df.sort_values('max.prop', ascending=False)
      df['ngram'] = df['entry'].str.count(' ') + 1
      df = df[df['ngram'].isin(self.ngrams)]
      # df = df.sort_values('ngram', ascending=False)
      # df['ngram'] = df['ngram'].str.replace('/[A-Z]+', '', regex=True)
      df.sort_values('entry', inplace=True)
      df.set_index('entry', inplace=True)
    else:
      df = pd.DataFrame()
      df.index.name = 'entry'
      
    self.original_lexicon = df
    self.lexicon = self.original_lexicon.copy()

  def get_original_lexicon(self):
    return self.original_lexicon

  def set_lexicon(self, min_freq=0, threshold=0.0):
    df = self.lexicon
    self.lexicon = df[(df['freq'] >= min_freq) & (df['max.prop'] > threshold)]
    self.min_freq = min_freq
    self.threshold = 0.0
  
  def get_lexicon(self):
    return self.lexicon
  
  def get(self, morph):
    return self.lexicon.loc[morph]
  
  def verify(self, morph, verbose=True):
    counts = self.lexicon.loc[morph, self.labels].astype('int')
    self.lexicon.loc[morph, 'freq'] = counts.sum()
    self.lexicon.loc[morph, 'max.value'] = counts.idxmax()
    self.lexicon.loc[morph, 'max.prop'] = counts.max() / counts.sum()
    if verbose:
      print(self.lexicon.loc[morph])

  def initialize_entry(self, morph, COMP=0, NEG=0, NEUT=0, None_=0, POS=0):
    row = pd.Series(dtype='object')
    row['ngram'] = morph.count(' ') + 1
    row['freq'] = 0
    row['COMP'] = COMP
    row['NEG'] = NEG
    row['NEUT'] = NEUT
    row['None'] = None_
    row['POS'] = POS
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

  def export_user_dict(self, DICT_PATH='./tagger/user_dictionary.txt'):
    unigrams = self.lexicon[self.lexicon['ngram'] == 1].index.tolist()
    with open(DICT_PATH, 'w') as f:
      f.writelines('\n'.join(['\t'.join(unigram.split('/')) for unigram in unigrams]))
    print('USER_DICT PATH:', DICT_PATH)
    self.dict_path = DICT_PATH

  def get_pattern(self, sorting=True):
    mylex = self.lexicon.copy()
    if sorting:
      sorts = sort(mylex)
    else:
      sorts = mylex
    
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
    return self.lexicon.apply(smooth, axis=1)

  def get_sent_probs(self, sentence, tagger, smoothing=True):
    matches = self.match(sentence, tagger)
    freqs = self.lexicon.loc[matches].copy()
    if smoothing:
      smoothed = freqs.apply(smooth, axis=1)
    else:
      smoothed = freqs

    return softmax(np.log(smoothed).sum()).sort_values(ascending=False)

class Tokenizer:
  def __init__(self, tagger_type='Komoran', dict_path='./tagger/user_dictionary.txt', model_name='snunlp/KR-ELECTRA-discriminator'):
    self.tagger_type = tagger_type

    if tagger_type == 'Okt':
      self.tagger = Okt()
    elif tagger_type == 'Mecab':
      self.tagger = Mecab()
    elif tagger_type == 'Kkma':
      self.tagger = Kkma()
    elif tagger_type == 'Komoran':
      self.tagger = Komoran(userdic=dict_path)
    elif tagger_type == 'Transformer':
      from transformers import AutoTokenizer
      self.tagger = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name)
  
  def tokenize(self, sentence):
    if self.tagger_type in ('Okt', 'Mecab', 'Kkma', 'Komoran'):
      tokens = self.tagger.pos(sentence, join=True)
    elif self.tagger_type == 'Transformer':
      tokens = self.tagger.tokenize(sentence)
    
    return tokens
  
  def get_tokens_str(self, sentence):
    return ' '.join(self.tokenize(sentence))

  def get_ngrams(self, sentence, ns):
    tokens = self.tokenize(sentence)
    return [' '.join(entry) for n in ns for entry in ngrams(tokens, n)]


