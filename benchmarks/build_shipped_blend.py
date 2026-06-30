"""Regenerate the shipped ``polarity-blend`` lexicon (src/kosac/data/).

The bundled artifact is the frozen KOSAC polarity seeds (counts ×alpha) summed
entry-wise with an NSMC-train-learned POS/NEG lexicon, written gzipped. The
default (min_freq=5, alpha=20) is the small shipped version; ``--full``
(min_freq=2) reproduces the strongest "champion" blend (~2.3 MB gz) from the
benchmark campaign — see ``benchmarks/README.md``.

Needs the Kiwi extra and downloads NSMC (CC0). Train-only, no test/NIKL leakage.

  python -m benchmarks.build_shipped_blend            # shipped (mf5, x20)
  python -m benchmarks.build_shipped_blend --full     # champion (mf2, x20)
"""
import argparse
import os

import pandas as pd

from kosac.lexicon import GenericLexicon, PolarityLexicon

from .build_nsmc import build as build_nsmc

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUT = os.path.join(REPO, 'src', 'kosac', 'data', 'polarity-blend.csv.gz')


def build(min_freq=5, alpha=20, ngrams=(1, 2, 3), out=DEFAULT_OUT):
  learned_csv = build_nsmc(pos_filter='all', min_freq=min_freq, ngrams=ngrams,
                           out_name=f'nsmc-train-all-mf{min_freq}')
  bundled = PolarityLexicon.load(ngrams=list(ngrams))
  learned = GenericLexicon(filepath=learned_csv, ngrams=list(ngrams))
  bp, bn = bundled.lexicon['POS'].astype(int), bundled.lexicon['NEG'].astype(int)
  lp, ln = learned.lexicon['POS'].astype(int), learned.lexicon['NEG'].astype(int)
  pos = (bp * alpha).round().astype(int).add(lp, fill_value=0).astype(int)
  neg = (bn * alpha).round().astype(int).add(ln, fill_value=0).astype(int)
  df = pd.DataFrame({'ngram': pos.index, 'NEG': neg.values, 'POS': pos.values})
  df[['ngram', 'NEG', 'POS']].to_csv(out, index=False, compression='gzip')
  print(f'wrote {out}: {len(df)} entries, {os.path.getsize(out) / 1e6:.2f} MB '
        f'(min_freq={min_freq}, alpha={alpha})')
  return out


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--min-freq', type=int, default=5)
  ap.add_argument('--alpha', type=float, default=20)
  ap.add_argument('--full', action='store_true', help='champion blend (min_freq=2)')
  ap.add_argument('--out', default=DEFAULT_OUT)
  args = ap.parse_args()
  build(min_freq=2 if args.full else args.min_freq, alpha=args.alpha, out=args.out)


if __name__ == '__main__':
  main()
