"""Re-optimize the blend ratio (alpha) and learned-lexicon min_freq UNDER the
multi-scale equal-weight scorer (the new champion scoring method).

Earlier alpha/min_freq sweeps used the greedy matcher; multi-scale changes the
optima (e.g. min_freq=10 helped greedy but hurts multi-scale). This refines both.

Run::  KOSAC_RUN_BENCH=1 python -m benchmarks.sweep_ms_refine
"""
import os

from kosac.lexicon import GenericLexicon

from . import run_eval, decision
from .build_blend import build
from .metrics import metrics, best_threshold

LEX = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'lexicons')
MF1 = os.path.join(LEX, 'nsmc-train-all-mf1.csv')
W = {1: 1, 2: 1, 3: 1}   # equal multi-scale weights (the sweep_ngram winner)
DS = None


def _learned_mf(mf):
  path = os.path.join(LEX, f'_ms_learned_mf{mf}.csv')
  if not os.path.exists(path):
    lex = GenericLexicon(filepath=MF1, ngrams=[1, 2, 3])
    lex.set_lexicon(min_freq=mf)
    lex.save(path)
  return path


def _score(csv):
  lex = GenericLexicon(filepath=csv, ngrams=[1, 2, 3])
  out = {}
  for d in DS:
    toks = run_eval.get_token_lists(d['name'], d['texts'], run_eval._plain_tokenizer(), 'plain')
    s, c = decision.score_logodds_multiscale(lex, toks, W)
    m = metrics('_', s, c, d['pos'], 0.0)
    m['obal'] = best_threshold(s, d['pos'])[1]
    out[d['name']] = m
  return out


def _row(tag, r):
  n, k = r['nsmc'], r['nikl']
  print(f'{tag:<18}{n["accuracy"]:>9.3f}{k["balanced_acc"]:>9.3f}{k["obal"]:>9.3f}'
        f'{k["roc_auc"]:>9.3f}{k["pr_auc_neg"]:>9.3f}')


def _header(title):
  print(f'\n### {title}')
  print(f'{"":18}{"NSMC acc":>9}{"NIKL bal":>9}{"orc bal":>9}{"NIKL ROC":>9}{"PR(NEG)":>9}')
  print('-' * 63)


def main():
  global DS
  DS = run_eval.load_datasets(('nsmc', 'nikl'))
  learned_mf2 = os.path.join(LEX, 'nsmc-train-all.csv')

  _header('(A) blend ratio alpha  [multi-scale, learned mf2]')
  for a in (0, 1, 5, 10, 20, 30, 40, 60, 100):
    csv = learned_mf2 if a == 0 else build(learned_mf2, alpha=a, name=f'_ms_blend_a{a:g}')
    _row(f'alpha={a:g}', _score(csv))

  _header('(B) learned min_freq  [multi-scale, no blend]')
  for mf in (1, 2, 3, 5, 10, 20):
    _row(f'min_freq={mf}', _score(_learned_mf(mf)))

  _header('(C) learned min_freq inside blend x20  [multi-scale]')
  for mf in (2, 3, 5, 10):
    csv = build(_learned_mf(mf), alpha=20, name=f'_ms_blend_mf{mf}_a20')
    _row(f'mf={mf} x20', _score(csv))


if __name__ == '__main__':
  main()
