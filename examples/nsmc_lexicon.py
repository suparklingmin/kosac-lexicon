"""Build a sentiment lexicon from a *new* corpus (NSMC) with the `kosac` API.

The bundled lexicons are frozen 2016 KOSAC data, but the same machinery builds a
lexicon from any labeled corpus. Here we use NSMC (Naver Sentiment Movie Corpus:
200k movie reviews labeled positive/negative) to learn a POS/NEG lexicon from
scratch, then inspect and score with it.

Requires the Kiwi extra (no Java):  pip install "kosac-lexicon[kiwi]"
Run:   python examples/nsmc_lexicon.py [--limit N] [--min-freq K]

NSMC (CC0) is downloaded from github.com/e9t/nsmc and cached in the system temp
directory; pass --no-download if you already have ratings_train.txt locally.
"""
import argparse
import os
import tempfile
import urllib.request

import pandas as pd

from kosac.corpora import Corpus
from kosac.lexicon import GenericLexicon

NSMC_URL = 'https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt'
# Content-word tags (Sejong/Kiwi): nouns, verbs, adjectives, roots, adverbs.
# Keeping only N-grams with a content token drops function-morpheme noise.
CONTENT = {'NNG', 'NNP', 'VV', 'VA', 'XR', 'MAG'}


def fetch_nsmc(cache_dir):
  path = os.path.join(cache_dir, 'ratings_train.txt')
  if not os.path.exists(path):
    print(f'downloading NSMC -> {path}')
    urllib.request.urlretrieve(NSMC_URL, path)
  return path


def to_corpus_csv(nsmc_path, out_path, limit=None):
  """NSMC is `id<TAB>document<TAB>label` (1=positive, 0=negative). Convert it to
  the headerless `text,label` CSV that `Corpus` reads, mapping 1->POS, 0->NEG."""
  df = pd.read_csv(nsmc_path, sep='\t', quoting=3, keep_default_na=False)
  df['document'] = df['document'].astype(str).str.strip()
  df = df[df['document'] != '']
  if limit:
    df = df.head(limit)
  df['label'] = df['label'].map({1: 'POS', 0: 'NEG'})
  df[['document', 'label']].to_csv(out_path, index=False, header=False)
  return out_path


def show_top(lex, label, k=12, min_freq=100):
  df = lex.get_lexicon()
  uni = df[(df['ngram'] == 1) & (df['freq'] >= min_freq) & (df['max.value'] == label)]
  top = uni.sort_values(['max.prop', 'freq'], ascending=False).head(k)
  print(f'\ntop {k} {label} unigrams (freq>={min_freq}):')
  for entry, row in top.iterrows():
    print(f'  {entry:<12} freq={int(row["freq"]):>5}  max.prop={row["max.prop"]:.3f}')


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--limit', type=int, default=None,
                  help='use only the first N reviews (faster; default: all 150k)')
  ap.add_argument('--ngrams', type=int, nargs='+', default=[1, 2, 3])
  ap.add_argument('--min-freq', type=int, default=5)
  ap.add_argument('--out', default='nsmc_lexicon.csv')
  args = ap.parse_args()

  try:
    from kosac.tokenizers import KiwiTokenizer
    tok = KiwiTokenizer()
  except ImportError as exc:
    print('[needs the Kiwi extra]', exc)
    return

  cache = os.path.join(tempfile.gettempdir(), 'kosac_nsmc')
  os.makedirs(cache, exist_ok=True)
  corpus_csv = to_corpus_csv(fetch_nsmc(cache),
                             os.path.join(cache, 'nsmc_corpus.csv'), args.limit)
  corpus = Corpus(corpus_csv)
  print(f'corpus: {len(corpus.df)} reviews, labels={corpus.get_labels()}')

  # Build the lexicon from the corpus -- the same call that rebuilds from any
  # labeled data. Vectorized counting makes the full 150k set a ~1-2 min job.
  lex = GenericLexicon(ngrams=args.ngrams)
  lex.set_labels(['POS', 'NEG'])
  print('building (tokenize + count)...')
  lex.update_from_corpus(corpus, tok, pos_tag=CONTENT, min_freq=args.min_freq)
  print(f'lexicon: {lex.get_size()} entries (min_freq={args.min_freq})')

  show_top(lex, 'POS')
  show_top(lex, 'NEG')

  # The package's CSV format joins morphemes with ';', so entries whose surface
  # is literally ';' (e.g. ';/SP') don't round-trip -- drop them before saving.
  clean = lex.get_lexicon()
  lex.lexicon = clean[~clean.index.str.contains(';', regex=False)]
  lex.save(args.out)
  print(f'\nsaved -> {args.out}  (reload with '
        f'GenericLexicon(filepath="{args.out}", ngrams={args.ngrams}))')

  # Score a few sentences. Shrink the match set first so the regex stays small.
  lex.set_lexicon(min_freq=max(args.min_freq, 50))
  print('\nsentence scoring:')
  for s in ['이 영화 정말 최고였다 여운이 남는다',
            '시간 낭비 최악의 영화 너무 지루하다',
            '연기는 좋은데 스토리가 좀 별로']:
    probs = lex.get_sent_probs(s, tok).round(3).to_dict()
    print(f'  {s}  ->  {probs}')


if __name__ == '__main__':
  main()
