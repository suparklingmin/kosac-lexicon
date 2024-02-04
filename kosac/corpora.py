import pandas as pd

class Corpus:
  # TODO: HuggingFace Datasets 라이브러리 추가하기
  def __init__(self, filepath='./data/example.csv'):
    self.df = pd.read_csv(filepath, names=('text', 'label'))
    self.labels = self.df['label'].unique().tolist()
  
  def get_labels(self):
    return self.labels
