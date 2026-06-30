"""Build cleaned/derived polarity lexicons from the frozen bundled CSV.

The frozen ``src/kosac/data/polarity.csv`` is NEVER modified; each variant is
written to ``benchmarks/results/lexicons/`` and evaluated via a ``{'csv': ...}``
config. Variants:

* ``polarity-nowild``  — drop the ~170 wildcard entries (``가*/JKS``) that are
  ``re.escape``d and can never match a Kiwi token (dead weight / documented bug).
* ``polarity-clean``   — nowild + drop ambiguous function-word entries (all
  morphemes are particles/endings AND max.prop < 0.5): out-of-domain noise.

Run::  python -m benchmarks.build_derived
"""
import os

from kosac.lexicon import PolarityLexicon

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, 'results', 'lexicons')

# Sejong function-morpheme tags: particles (J*), endings (E*), and the copula.
FUNCTION_TAGS = {
    'JKS', 'JKC', 'JKG', 'JKO', 'JKB', 'JKV', 'JKQ', 'JX', 'JC',
    'EP', 'EF', 'EC', 'ETN', 'ETM', 'VCP', 'VCN',
}


def _tag(token):
  return token.rsplit('/', 1)[-1]


def _all_function(entry):
  return all(_tag(tok) in FUNCTION_TAGS for tok in entry.split(' '))


def _save(lex, df, name):
  os.makedirs(OUT_DIR, exist_ok=True)
  path = os.path.join(OUT_DIR, f'{name}.csv')
  lex.lexicon = df
  lex.save(path)
  print(f'  {name}: {len(df)} entries -> {path}')
  return path


def build():
  lex = PolarityLexicon.load(ngrams=[1, 2, 3])
  full = lex.lexicon
  print(f'bundled polarity: {len(full)} entries')

  # nowild: drop literal-'*' entries (never match after re.escape).
  is_wild = full.index.to_series().str.contains('*', regex=False)
  nowild = full[~is_wild]
  print(f'  dropped {int(is_wild.sum())} wildcard entries')
  _save(lex, nowild, 'polarity-nowild')

  # clean: also drop ambiguous all-function entries (max.prop < 0.5).
  ambiguous_fn = nowild.index.to_series().map(_all_function) & (nowild['max.prop'] < 0.5)
  clean = nowild[~ambiguous_fn]
  print(f'  dropped {int(ambiguous_fn.sum())} ambiguous function-word entries')
  _save(lex, clean, 'polarity-clean')

  # nofunc: drop ALL function-only UNIGRAMS (regardless of max.prop) — a bare
  # particle/ending carries no standalone polarity, so it is pure noise. N-grams
  # that merely contain a function morpheme (e.g. 'ㄹ/ETM 수/NNB 있/VV') are kept.
  is_func_unigram = (nowild['ngram'] == 1) & nowild.index.to_series().map(
      lambda e: _tag(e) in FUNCTION_TAGS)
  nofunc = nowild[~is_func_unigram]
  print(f'  dropped {int(is_func_unigram.sum())} function-only unigrams')
  _save(lex, nofunc, 'polarity-nofunc')


if __name__ == '__main__':
  build()
