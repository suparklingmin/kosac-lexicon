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

  def tokenize_batch(self, sentences):
    """Tokenize many sentences at once. The default applies :meth:`tokenize`
    per sentence; tokenizers with a batch API (e.g. Kiwi) override this to
    process the whole list in parallel."""
    return [self.tokenize(sentence) for sentence in sentences]

  def get_ngrams_batch(self, sentences, ns):
    """N-grams for many sentences (one list per sentence), via tokenize_batch.

    Equivalent to ``[get_ngrams(s, ns) for s in sentences]`` but routed through
    :meth:`tokenize_batch`, so a batch tokenizer parallelizes the heavy step."""
    return [[' '.join(entry) for n in ns for entry in ngrams(tokens, n)]
            for tokens in self.tokenize_batch(sentences)]


# Kiwi (kiwipiepy)
class KiwiTokenizer(Tokenizer):
  """Morpheme tokenizer backed by Kiwi. Requires ``kosac-lexicon[kiwi]``.

  Emits ``surface/POS`` tokens (e.g. ``힘/NNG``) matching the lexicon's entry
  format. Extra keyword arguments are forwarded to ``kiwipiepy.Kiwi``.
  """

  def __init__(self, user_words=None, **kwargs):
    try:
      from kiwipiepy import Kiwi
    except ImportError as exc:
      raise ImportError(
          'KiwiTokenizer requires `pip install kosac-lexicon[kiwi]`.'
      ) from exc
    self.kiwi = Kiwi(**kwargs)
    if user_words:
      self.add_user_words(user_words)

  @staticmethod
  def _split_word(word):
    if isinstance(word, (tuple, list)):
      return word[0], word[1]
    form, _sep, tag = str(word).rpartition('/')
    return form, tag

  def add_user_words(self, words, score=0.0):
    """Register ``(form, tag)`` pairs (or ``'form/tag'`` strings) as Kiwi user
    words. Returns the number successfully added; entries Kiwi rejects (e.g. the
    wildcard ``*`` forms) are skipped."""
    added = 0
    for word in words:
      form, tag = self._split_word(word)
      if not form or not tag:
        continue
      try:
        self.kiwi.add_user_word(form, tag, score)
        added += 1
      except Exception:
        continue
    return added

  @classmethod
  def from_lexicon(cls, lexicon, tags=None, score=0.0, **kwargs):
    """Build a tokenizer seeded with a lexicon's unigram entries as user words.

    This biases Kiwi toward segmenting text the way the lexicon expects, easing
    the Sejong/Kiwi tagset mismatch. ``tags`` optionally restricts which POS
    tags are registered (e.g. ``{'NNG', 'VV', 'VA', 'XR'}`` for content words).
    """
    tokenizer = cls(**kwargs)
    unigrams = lexicon.get_lexicon()
    entries = unigrams[unigrams['ngram'] == 1].index
    words = [(form, tag) for form, tag in (cls._split_word(e) for e in entries)
             if tags is None or tag in tags]
    tokenizer.add_user_words(words, score=score)
    return tokenizer

  def tokenize(self, sentence):
    return [f'{token.form}/{token.tag}' for token in self.kiwi.tokenize(sentence)]

  def tokenize_batch(self, sentences):
    # Kiwi tokenizes an iterable of sentences in parallel; results stay in order
    # and are identical to tokenizing each sentence on its own.
    return [[f'{token.form}/{token.tag}' for token in tokens]
            for tokens in self.kiwi.tokenize(list(sentences))]

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
