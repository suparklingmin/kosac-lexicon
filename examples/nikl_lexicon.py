"""Build a 3-class polarity lexicon from scored sentiment *expressions*.

Some sentiment corpora label short *expressions* (phrases) with a numeric
polarity score instead of a whole-document label. A common shape -- e.g. the
NIKL / 모두의 말뭉치 sentiment-analysis corpus -- is a data frame with an
``expression_form`` column (the phrase) and an integer ``expression_score`` in
-2..2.  This script maps the score to three classes and builds a morpheme
polarity lexicon with the standard `kosac` machinery:

    score  2, 1  -> POS      (positive)
    score     0  -> NEUT     (neutral)
    score -1, -2 -> NEG      (negative)

Unlike the binary NSMC example, this yields a **3-class** lexicon (POS/NEUT/NEG),
the same label set as KOSAC's own polarity feature minus COMP/None.

Requires the Kiwi extra (no Java):  pip install "kosac-lexicon[kiwi]"
Run:   python examples/nikl_lexicon.py [--data FILE.csv]
                                       [--min-freq K] [--ngrams 1 2 3]
                                       [--pos-filter content|content+ic|all]

With no ``--data`` a tiny built-in sample of invented expressions is used so the
script runs out of the box.  Point ``--data`` at your own CSV that has
``expression_form`` and ``expression_score`` columns to build a real lexicon --
e.g. a 모두의 말뭉치 sentiment export::

    python examples/nikl_lexicon.py --data EXSA2002108040.csv

Any other columns are ignored, so a raw corpus export works directly.
"""
import argparse
import tempfile
import os

import pandas as pd

from kosac.corpora import Corpus
from kosac.lexicon import GenericLexicon

# The 3 polarity classes, in column order. KOSAC's polarity feature is the same
# set plus COMP/None; here the source only distinguishes positive/neutral/negative.
LABELS = ['POS', 'NEUT', 'NEG']

# Content-word tags (Sejong/Kiwi): nouns, verbs, adjectives, roots, adverbs.
# Keeping only N-grams with a content token drops function-morpheme noise and
# gives a clean, readable lexicon. `all` keeps every token (endings, particles,
# punctuation carry sentiment too) and maximizes coverage when scoring.
POS_FILTERS = {
    'content': {'NNG', 'NNP', 'VV', 'VA', 'XR', 'MAG'},
    'content+ic': {'NNG', 'NNP', 'VV', 'VA', 'XR', 'MAG', 'IC'},
    'all': None,
}

# A tiny EXSA-shaped sample (expression_form, expression_score) used when no
# --data file is given, so the script runs with no external corpus. These
# expressions are invented for the demo, not taken from any corpus.
SAMPLE = pd.DataFrame({
    'expression_form': [
        '정말 마음에 쏙 들어요', '완전 최고예요', '생각보다 괜찮네요', '무난하게 좋아요',
        '쓸 만한 것 같아요', '재구매 의사 있어요', '배송도 빠르고 좋아요', '품질이 만족스러워요',
        '그냥 보통이에요', '잘 모르겠어요', '조금 아쉬워요', '기대보다는 별로네요',
        '다시 살지는 모르겠어요', '향이 좀 강해요', '완전 실망했어요', '돈이 너무 아까워요',
        '두 번 다시 안 사요', '품질이 최악이에요',
    ],
    'expression_score': [2, 2, 1, 1, 1, 1, 1, 1, 0, 0, -1, -1, -1, -1, -2, -2, -2, -2],
})


def score_to_label(score):
  """Map an integer expression score to a polarity class."""
  if score > 0:
    return 'POS'
  if score < 0:
    return 'NEG'
  return 'NEUT'


def to_corpus_csv(df, out_path):
  """Turn an expression-scored data frame into the headerless `text,label` CSV
  that `Corpus` reads: keep the two columns, drop blanks, map score -> label."""
  df = df[['expression_form', 'expression_score']].copy()
  df['expression_form'] = df['expression_form'].astype(str).str.strip()
  df = df[df['expression_form'] != '']
  df['label'] = df['expression_score'].astype(int).map(score_to_label)
  df[['expression_form', 'label']].to_csv(out_path, index=False, header=False)
  return out_path


def show_top(lex, label, k=10, min_freq=3):
  df = lex.get_lexicon()
  uni = df[(df['ngram'] == 1) & (df['freq'] >= min_freq) & (df['max.value'] == label)]
  top = uni.sort_values(['max.prop', 'freq'], ascending=False).head(k)
  print(f'\ntop {k} {label} unigrams (freq>={min_freq}):')
  for entry, row in top.iterrows():
    print(f'  {entry:<14} freq={int(row["freq"]):>5}  max.prop={row["max.prop"]:.3f}')


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--data', default=None,
                  help='CSV with expression_form + expression_score columns '
                       '(default: a small built-in sample)')
  ap.add_argument('--ngrams', type=int, nargs='+', default=[1, 2, 3])
  ap.add_argument('--min-freq', type=int, default=2)
  ap.add_argument('--pos-filter', choices=sorted(POS_FILTERS), default='content',
                  help="'content' for a clean lexicon (default); 'all' for max coverage")
  ap.add_argument('--out', default='nikl_lexicon.csv')
  args = ap.parse_args()
  pos_tag = POS_FILTERS[args.pos_filter]

  try:
    from kosac.tokenizers import KiwiTokenizer
    tok = KiwiTokenizer()
  except ImportError as exc:
    print('[needs the Kiwi extra]', exc)
    return

  raw = pd.read_csv(args.data) if args.data else SAMPLE
  source = args.data or 'built-in sample (pass --data for a real corpus)'
  print(f'source: {source}  ({len(raw)} expressions)')
  print('score -> label counts:',
        raw['expression_score'].astype(int).map(score_to_label).value_counts().to_dict())

  cache = os.path.join(tempfile.gettempdir(), 'kosac_expressions')
  os.makedirs(cache, exist_ok=True)
  corpus = Corpus(to_corpus_csv(raw, os.path.join(cache, 'expr_corpus.csv')))

  # Build the 3-class lexicon -- the same call that rebuilds from any labeled
  # corpus, here with three labels instead of NSMC's two.
  lex = GenericLexicon(ngrams=args.ngrams)
  lex.set_labels(LABELS)
  print(f'building (tokenize + count, pos-filter={args.pos_filter})...')
  lex.update_from_corpus(corpus, tok, pos_tag=pos_tag, min_freq=args.min_freq)
  print(f'lexicon: {lex.get_size()} entries (min_freq={args.min_freq})')

  show_top(lex, 'POS')
  show_top(lex, 'NEG')

  lex.save(args.out)
  print(f'\nsaved -> {args.out}  (reload with '
        f'GenericLexicon(filepath="{args.out}", ngrams={args.ngrams}))')

  # Score a few sentences. softmax over the matched morphemes' smoothed log-probs.
  print('\nsentence scoring:')
  for s in ['이 제품 정말 만족스럽고 좋다', '돈 아깝고 최악이라 다시는 안 산다',
            '향은 신선한데 포장이 좀 별로']:
    probs = lex.get_sent_probs(s, tok).round(3).to_dict()
    print(f'  {s}  ->  {probs}')


if __name__ == '__main__':
  main()
