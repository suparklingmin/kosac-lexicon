import pandas as pd
import numpy as np
from konlpy.tag import Komoran
import re

def get_lexicon(filepath='./lexicon/polarity.csv', min_freq=0, threshold=0.0):
  lexicon = pd.read_csv(filepath)
  lexicon['entry'] = lexicon['ngram'].str.replace(';', ' ')
  lexicon['COMP'] = (lexicon['COMP'] * lexicon['freq']).astype('int')
  lexicon['NEG'] = (lexicon['NEG'] * lexicon['freq']).astype('int')
  lexicon['NEUT'] = (lexicon['NEUT'] * lexicon['freq']).astype('int')
  lexicon['None'] = (lexicon['None'] * lexicon['freq']).astype('int')
  lexicon['POS'] = (lexicon['POS'] * lexicon['freq']).astype('int')
  lexicon = lexicon[~lexicon['ngram'].str.contains('*/', regex=False)]
  lexicon = lexicon.sort_values('max.prop', ascending=False)
  lexicon['ngram'] = lexicon['entry'].str.count(' ') + 1
  lexicon = lexicon.sort_values('ngram', ascending=False)
  # lexicon['ngram'] = lexicon['ngram'].str.replace('/[A-Z]+', '', regex=True)
  lexicon.set_index('entry', inplace=True)

  selected_lexicon = lexicon[(lexicon['freq'] >= min_freq) & (lexicon['max.prop'] > threshold)]
  selected_lexicon

  return selected_lexicon

def add_one(x:np.array):
    return (x+1)/(x+1).sum()

def smooth(lexicon, func=add_one):
  counts = lexicon[['COMP', 'NEG', 'NEUT', 'None', 'POS']]
  counts = counts.apply(add_one, axis=1)
  return counts

def update_dict(lexicon, DICT_PATH='./tagger/user_dictionary.txt'):
  unigrams = lexicon[lexicon['ngram'] == 1].index.tolist()
  with open(DICT_PATH, 'w') as f:
    f.writelines('\n'.join(['\t'.join(unigram.split('/')) for unigram in unigrams]))
  print('USERDIC:', DICT_PATH)

def get_tagger(DICT_PATH='./tagger/user_dictionary.txt'):
   return Komoran(userdic=DICT_PATH)

def tokenize(sentence, tagger):
  tagger.pos(sentence)
  tagged = ' '.join(tagger.pos(sentence, join=True))
  return tagged

def match(sentence, tagger, lexicon):
  tagged = tokenize(sentence, tagger)
  pattern = re.compile('|'.join(lexicon.index))
  matches = pattern.findall(tagged)
  return matches

def get_match_info(sentence, tagger, lexicon):
  matches = match(sentence, tagger, lexicon)
  result = [(match, lexicon.loc[match]['max.value'], lexicon.loc[match]['max.prop']) for match in matches]
  return result

def softmax(x:np.array):
  x = x - x.max()
  return np.exp(x) / sum(np.exp(x))

def get_sent_probs(sentence, tagger, lexicon):
  matches = match(sentence, tagger, lexicon)
  smoothed = smooth(lexicon)
  return softmax(np.log(smoothed.loc[matches]).sum())