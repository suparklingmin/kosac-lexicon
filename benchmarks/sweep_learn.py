"""Sweep the learning-side levers of the NSMC-train lexicon.

Three knobs, each evaluated on NSMC (in-domain) and NIKL (out-of-domain):
  (A) min_freq  — entry frequency cutoff (applied as a set_lexicon filter on a
                  min_freq=1 superset, so no rebuild per value)
  (B) ngrams    — match unigrams / +bigrams / +trigrams (eval-time match length)
  (C) pos-filter— which Sejong tags an n-gram must contain (build-time; one build
                  per filter)

NIKL ROC-AUC / PR-AUC(NEG) are the finer-grained OOD signals (the 58-NEG set makes
balanced-acc coarse); NSMC accuracy is the in-domain objective.

Run::  KOSAC_RUN_BENCH=1 python -m benchmarks.sweep_learn
"""
import os

from . import run_eval
from .build_nsmc import build

LEX = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'lexicons')
DS = None


def _ensure(pos_filter):
  path = os.path.join(LEX, f'nsmc-train-{pos_filter}-mf1.csv')
  if not os.path.exists(path):
    build(pos_filter=pos_filter, min_freq=1, out_name=f'nsmc-train-{pos_filter}-mf1')
  return path


def _ev(csv, ngrams, min_freq):
  cfg = dict(name='_', lexicon={'csv': csv}, ngrams=ngrams, rule='logodds',
             min_freq=min_freq, oracle=False)
  return {d['name']: run_eval.evaluate(cfg, d)[0] for d in DS}


def _row(tag, r):
  n, k = r['nsmc'], r['nikl']
  print(f'{tag:<22}{n["accuracy"]:>9.3f}{k["balanced_acc"]:>9.3f}'
        f'{k["roc_auc"]:>9.3f}{k["pr_auc_neg"]:>9.3f}{n["coverage"]:>9.3f}')


def _header(title):
  print(f'\n### {title}')
  print(f'{"":22}{"NSMC acc":>9}{"NIKL bal":>9}{"NIKL ROC":>9}{"NIKL PRn":>9}'
        f'{"NSMC cov":>9}')
  print('-' * 67)


def main():
  global DS
  DS = run_eval.load_datasets(('nsmc', 'nikl'))
  all1 = _ensure('all')
  cont1 = _ensure('content')
  ci1 = _ensure('content+ic')

  _header('(A) min_freq sweep  [pos=all, ngrams=1,2,3]')
  for mf in (1, 2, 3, 5, 10, 20):
    _row(f'min_freq={mf}', _ev(all1, [1, 2, 3], mf))

  _header('(B) ngram sweep  [pos=all, min_freq=2]')
  for ng in ([1], [1, 2], [1, 2, 3]):
    _row(f'ngrams={ng}', _ev(all1, ng, 2))

  _header('(C) pos-filter  [min_freq=2, ngrams=1,2,3]')
  for name, csv in (('all', all1), ('content', cont1), ('content+ic', ci1)):
    _row(f'pos={name}', _ev(csv, [1, 2, 3], 2))


if __name__ == '__main__':
  main()
