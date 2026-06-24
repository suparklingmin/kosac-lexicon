"""Minimal end-to-end example for the `kosac` package.

Run with the core install (no Java needed) for the data-only parts:

    python examples/quickstart.py

The tagger-based section additionally requires `pip install "kosac-lexicon[kiwi]"`
(no Java needed); it is skipped automatically when Kiwi is not available.
"""
import kosac


def main():
  print('available features:', kosac.FEATURES)
  print('data version:', kosac.__data_version__)

  # Load the polarity lexicon (unigrams, keep entries seen in >= 5 Seeds).
  lex = kosac.load_lexicon('polarity', ngrams=[1], min_freq=5)
  print('polarity unigram entries (min_freq=5):', lex.get_size())
  print(lex.get_entry('힘/NNG'))

  # Tagging + sentence-level scoring needs a tokenizer (optional extra).
  try:
    from kosac.tokenizers import KiwiTokenizer
    tok = KiwiTokenizer()
  except ImportError as exc:
    print('\n[skipping tagger demo]', exc)
    return

  lex = kosac.load_lexicon('polarity', ngrams=[1, 2, 3])
  sentence = '나는 정말 행복하다'
  print('\nmatches:', lex.match_patterns(sentence, tok))
  print('sentiment probabilities:')
  print(lex.get_sent_probs(sentence, tok))


if __name__ == '__main__':
  main()
