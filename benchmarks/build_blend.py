"""Build a union-blended POS/NEG lexicon (bundled KOSAC + an NSMC-learned CSV).

Collapses the bundled 5-label polarity lexicon to its POS/NEG counts and adds
them entry-wise to a learned POS/NEG lexicon, then re-derives the absolute-count
CSV. The idea: bundled contributes curated coverage, the learned one contributes
domain signal. (A score-level ensemble — usually cleaner — needs no build step;
use an ``{'ensemble': [...]}`` config instead.)

Note: blending the CC BY-SA bundled data makes the artifact CC BY-SA.

Run::  python -m benchmarks.build_blend --learned results/lexicons/nsmc-train-all.csv
"""
import argparse
import os

import pandas as pd

from kosac.lexicon import PolarityLexicon, GenericLexicon

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'lexicons')


def _pos_neg_counts(lex):
  df = lex.lexicon
  return df['POS'].astype(int), df['NEG'].astype(int)


def build(learned_csv, ngrams=(1, 2, 3), alpha=1.0, bundled_csv=None, name=None):
  """Union-blend bundled ⊕ learned POS/NEG counts.

  ``alpha`` scales the bundled counts before summing — the blend "ratio" knob.
  Bundled (KOSAC seed) counts are tiny next to NSMC frequencies, so at alpha=1 the
  blend ≈ the learned lexicon; raising alpha gives the curated frozen data more
  weight. ``bundled_csv`` blends a derived variant instead of the frozen lexicon.
  """
  if bundled_csv:
    bundled = GenericLexicon(filepath=bundled_csv, ngrams=list(ngrams))
  else:
    bundled = PolarityLexicon.load(ngrams=list(ngrams))
  learned = GenericLexicon(filepath=learned_csv, ngrams=list(ngrams))

  bp, bn = _pos_neg_counts(bundled)
  lp, ln = _pos_neg_counts(learned)
  pos = (bp * alpha).round().astype(int).add(lp, fill_value=0)
  neg = (bn * alpha).round().astype(int).add(ln, fill_value=0)
  blended = pd.DataFrame({'ngram': pos.index, 'POS': pos.astype(int),
                          'NEG': neg.astype(int)})

  os.makedirs(OUT_DIR, exist_ok=True)
  if name is None:
    stem = os.path.splitext(os.path.basename(learned_csv))[0]
    name = f'blend-{stem}' if alpha == 1.0 else f'blend-{stem}-a{alpha:g}'
  path = os.path.join(OUT_DIR, f'{name}.csv')
  blended[['ngram', 'POS', 'NEG']].to_csv(path, index=False)
  print(f'  blend(alpha={alpha:g}): {len(blended)} entries -> {path}')
  return path


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--learned', required=True, help='learned POS/NEG lexicon CSV')
  ap.add_argument('--ngrams', type=int, nargs='+', default=[1, 2, 3])
  ap.add_argument('--alpha', type=float, default=1.0,
                  help='bundled count multiplier (blend ratio)')
  ap.add_argument('--bundled', default=None,
                  help='use this CSV as the bundled side (default: frozen lexicon)')
  args = ap.parse_args()
  build(args.learned, tuple(args.ngrams), alpha=args.alpha, bundled_csv=args.bundled)


if __name__ == '__main__':
  main()
