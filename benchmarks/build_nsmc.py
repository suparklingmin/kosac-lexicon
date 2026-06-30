"""Learn a POS/NEG lexicon from the NSMC TRAIN split (leakage-safe).

Built only from ``ratings_train.txt``; the NSMC test split and NIKL are never
seen here, so the resulting lexicon can be evaluated on both without leakage. The
train file's SHA-256 is printed for the audit trail. Writes to
``benchmarks/results/lexicons/nsmc-train-<filter>.csv``.

Run::  python -m benchmarks.build_nsmc --pos-filter all --min-freq 2
"""
import argparse
import os

from kosac.lexicon import GenericLexicon

from . import datasets

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'lexicons')

# Same content-word tagset as examples/nsmc_lexicon.py.
POS_FILTERS = {
    'content': {'NNG', 'NNP', 'VV', 'VA', 'XR', 'MAG'},
    'content+ic': {'NNG', 'NNP', 'VV', 'VA', 'XR', 'MAG', 'IC'},
    'all': None,
}


def build(pos_filter='all', min_freq=2, ngrams=(1, 2, 3), out_name=None):
  from kosac.tokenizers import KiwiTokenizer
  corpus, info = datasets.load_nsmc_train()
  print(f'NSMC train: {info["rows"]} reviews (sha={info["sha256"][:12]}) '
        f'-- TRAIN ONLY, no test/NIKL leakage')
  tok = KiwiTokenizer()
  lex = GenericLexicon(ngrams=list(ngrams))
  lex.set_labels(['POS', 'NEG'])
  print(f'building (pos-filter={pos_filter}, min_freq={min_freq})...')
  lex.update_from_corpus(corpus, tok, pos_tag=POS_FILTERS[pos_filter], min_freq=min_freq)
  os.makedirs(OUT_DIR, exist_ok=True)
  path = os.path.join(OUT_DIR, f'{out_name or f"nsmc-train-{pos_filter}"}.csv')
  lex.save(path)
  print(f'  {lex.get_size()} entries -> {path}')
  return path


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--pos-filter', choices=sorted(POS_FILTERS), default='all')
  ap.add_argument('--min-freq', type=int, default=2)
  ap.add_argument('--ngrams', type=int, nargs='+', default=[1, 2, 3])
  args = ap.parse_args()
  build(args.pos_filter, args.min_freq, tuple(args.ngrams))


if __name__ == '__main__':
  main()
