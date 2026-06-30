"""Fine sweep of the union blend ratio (bundled-count multiplier alpha).

For each alpha, builds the bundled-x-alpha ⊕ NSMC-train union blend and evaluates
it on NSMC (in-domain) and NIKL (out-of-domain). NIKL balanced-acc is the primary
objective but is coarse (only ~58 NEG docs, so ~1.7pp per doc); ROC-AUC and
PR-AUC(NEG) use the full ranking and are the finer-grained tie-breakers.

Run::  KOSAC_RUN_BENCH=1 python -m benchmarks.sweep_alpha 8 10 12 15 20 25 30 40
"""
import os
import sys

from . import run_eval
from .build_blend import build

NSMC = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    'results', 'lexicons', 'nsmc-train-all.csv')


def main():
  alphas = [float(a) for a in sys.argv[1:]] or [1, 5, 10, 15, 20, 30, 50, 100]
  ds = run_eval.load_datasets(('nsmc', 'nikl'))
  print(f'\n{"alpha":>6}{"NSMC acc":>9}{"NIKL bal":>9}{"NIKL ROC":>9}'
        f'{"NIKL PRn":>9}{"NEG rec":>10}')
  print('-' * 52)
  best = (None, -1.0)
  for a in alphas:
    path = build(NSMC, alpha=a, name=f'blend-sweep-a{a:g}')
    cfg = dict(name=f'U-a{a:g}', lexicon={'csv': path}, ngrams=[1, 2, 3], rule='logodds')
    rows = {d['name']: run_eval.evaluate(cfg, d)[0] for d in ds}
    n, k = rows['nsmc'], rows['nikl']
    print(f'{a:>6g}{n["accuracy"]:>9.3f}{k["balanced_acc"]:>9.3f}'
          f'{k["roc_auc"]:>9.3f}{k["pr_auc_neg"]:>9.3f}'
          f'{k["neg_recall_n"]:>6d}/{k["neg_total"]:<3d}')
    if k['balanced_acc'] > best[1]:
      best = (a, k['balanced_acc'])
  print(f'\nNIKL balanced-acc peak: alpha={best[0]:g} (bal={best[1]:.3f})')


if __name__ == '__main__':
  main()
