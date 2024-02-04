import pandas as pd
import numpy as np

# Smoothing
def add_one(x:np.array):
  return (x+1)/(x+1).sum()

def smooth(row:pd.Series, labels:list, func=add_one):
  counts = row[labels].astype('int')
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

