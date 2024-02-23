from nltk import ngrams
from konlpy.tag import Komoran, Kkma, Okt
from transformers import AutoTokenizer

class Tokenizer:
  def tokenize(self, sentence):
    return sentence.split()
  
  def get_tokens_str(self, sentence):
    return ' '.join(self.tokenize(sentence))

  def get_ngrams(self, sentence, ns):
    tokens = self.tokenize(sentence)
    return [' '.join(entry) for n in ns for entry in ngrams(tokens, n)]


# KoNLPy
class PosTokenizer(Tokenizer):
  tagger_type = None

  def __init__(self, **kwargs):
    if self.tagger_type:
      self.tagger = self.tagger_type(**kwargs)

  def tokenize(self, sentence):
    tokens = self.tagger.pos(sentence, join=True)
    return tokens

class KomoranTokenizer(PosTokenizer):
  tagger_type = Komoran

class KkmaTokenizer(PosTokenizer):
  tagger_type = Kkma

class OktTokenizer(PosTokenizer):
  tagger_type = Okt

# HuggingFace Transformers
class HuggingFaceTokenizer(Tokenizer, AutoTokenizer):
  def __init__(self, model_name='snunlp/KR-ELECTRA-discriminator'):
    tokenizer = self.from_pretrained(pretrained_model_name_or_path=model_name)
    self.tagger_type = type(tokenizer)
    self = tokenizer
  