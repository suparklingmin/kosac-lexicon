"""Tokenizers used to turn raw Korean text into ``surface/POS`` morpheme tokens.

The base :class:`Tokenizer` is pure-Python. :class:`KiwiTokenizer` (the
recommended POS tokenizer) requires ``kosac-lexicon[kiwi]`` — Kiwi installs from
PyPI with no Java/JVM. :class:`HuggingFaceTokenizer` requires
``kosac-lexicon[transformers]``. Those heavy dependencies are imported lazily so
that ``import kosac`` never fails when the extras are absent.

Note on tagsets: the lexicon entries use Sejong-style ``surface/POS`` tokens.
Kiwi's tagset is Sejong-based and aligns on the common content/function morpheme
tags (NNG, VV, VA, JKS, EC, ...), so it matches the lexicon well; a few symbol
and web tags differ from the original tagger and may segment differently.
"""
from .utils import ngrams


class Tokenizer:
  """Whitespace tokenizer and n-gram helper. No external dependencies."""

  def tokenize(self, sentence):
    return sentence.split()

  def get_tokens_str(self, sentence):
    return ' '.join(self.tokenize(sentence))

  def tokenize_with_offsets(self, sentence):
    """Return ``(token, start, end)`` triples aligning tokens to char offsets.

    The default implementation locates each token as a substring of the input.
    Subclasses with real offset information (e.g. Kiwi) should override it.
    """
    spans = []
    cursor = 0
    for token in self.tokenize(sentence):
      start = sentence.find(token, cursor)
      if start < 0:
        start = cursor
      end = start + len(token)
      spans.append((token, start, end))
      cursor = end
    return spans

  def get_ngrams(self, sentence, ns):
    tokens = self.tokenize(sentence)
    return [' '.join(entry) for n in ns for entry in ngrams(tokens, n)]


# Kiwi (kiwipiepy)
class KiwiTokenizer(Tokenizer):
  """Morpheme tokenizer backed by Kiwi. Requires ``kosac-lexicon[kiwi]``.

  Emits ``surface/POS`` tokens (e.g. ``힘/NNG``) matching the lexicon's entry
  format. Extra keyword arguments are forwarded to ``kiwipiepy.Kiwi``.
  """

  def __init__(self, **kwargs):
    try:
      from kiwipiepy import Kiwi
    except ImportError as exc:
      raise ImportError(
          'KiwiTokenizer requires `pip install kosac-lexicon[kiwi]`.'
      ) from exc
    self.kiwi = Kiwi(**kwargs)

  def tokenize(self, sentence):
    return [f'{token.form}/{token.tag}' for token in self.kiwi.tokenize(sentence)]

  def tokenize_with_offsets(self, sentence):
    return [(f'{token.form}/{token.tag}', token.start, token.start + token.len)
            for token in self.kiwi.tokenize(sentence)]


# HuggingFace Transformers
class HuggingFaceTokenizer(Tokenizer):
  """Subword tokenizer backed by a HuggingFace model. Requires ``kosac-lexicon[transformers]``."""

  def __init__(self, model_name='snunlp/KR-ELECTRA-discriminator'):
    try:
      from transformers import AutoTokenizer
    except ImportError as exc:
      raise ImportError(
          'HuggingFace tokenizer requires `pip install kosac-lexicon[transformers]`.'
      ) from exc
    self.tagger = AutoTokenizer.from_pretrained(model_name)

  def tokenize(self, sentence):
    return self.tagger.tokenize(sentence)
