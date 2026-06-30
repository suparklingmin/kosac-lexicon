"""Sweep per-length n-gram weights for the multi-scale log-odds scorer.

Background: with the greedy non-overlapping matcher, unigrams-only rank best
out-of-domain (NIKL ROC 0.724) while [1,2,3] is best in-domain — a trade-off,
because a matched trigram suppresses its unigrams. The multi-scale scorer
(`decision.score_logodds_multiscale`) scores every length independently and
combines them with weights, so we can keep the unigram base and *add* down-
weighted higher orders. This sweep looks for a weighting that holds the unigram
OOD ranking while recovering in-domain accuracy / PR(NEG).

Run::  KOSAC_RUN_BENCH=1 python -m benchmarks.sweep_ngram
"""
import os

from kosac.lexicon import GenericLexicon

from . import run_eval, decision
from .metrics import metrics

LEX = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'lexicons')

# (label, {length: weight})
PRESETS = [
    ('uni only          [1,0,0]', {1: 1, 2: 0, 3: 0}),
    ('bi only           [0,1,0]', {1: 0, 2: 1, 3: 0}),
    ('tri only          [0,0,1]', {1: 0, 2: 0, 3: 1}),
    ('equal             [1,1,1]', {1: 1, 2: 1, 3: 1}),
    ('uni+bi            [1,1,0]', {1: 1, 2: 1, 3: 0}),
    ('decay .5          [1,.5,.25]', {1: 1, 2: 0.5, 3: 0.25}),
    ('decay .3          [1,.3,.1]', {1: 1, 2: 0.3, 3: 0.1}),
    ('uni-heavy         [2,1,.5]', {1: 2, 2: 1, 3: 0.5}),
    ('uni-strong        [3,1,.3]', {1: 3, 2: 1, 3: 0.3}),
]


def _eval_scores(scorer, lex, ds):
  out = {}
  for d in ds:
    toks = run_eval.get_token_lists(d['name'], d['texts'], run_eval._plain_tokenizer(), 'plain')
    scores, counts = scorer(lex, toks)
    out[d['name']] = metrics('_', scores, counts, d['pos'], threshold=0.0)
  return out


def main(lexicon_csv=None):
  lexicon_csv = lexicon_csv or os.path.join(LEX, 'nsmc-train-all.csv')
  print(f'lexicon: {os.path.basename(lexicon_csv)}')
  lex = GenericLexicon(filepath=lexicon_csv, ngrams=[1, 2, 3])
  ds = run_eval.load_datasets(('nsmc', 'nikl'))

  print(f'\n{"weights":<26}{"NSMC acc":>9}{"NIKL bal":>9}{"NIKL ROC":>9}{"NIKL PRn":>9}')
  print('-' * 62)
  # reference: greedy leftmost-longest [1,2,3]
  ref = _eval_scores(lambda l, t: decision.score_logodds(l, t, [1, 2, 3]), lex, ds)
  print(f'{"greedy [1,2,3] (ref)":<26}{ref["nsmc"]["accuracy"]:>9.3f}'
        f'{ref["nikl"]["balanced_acc"]:>9.3f}{ref["nikl"]["roc_auc"]:>9.3f}'
        f'{ref["nikl"]["pr_auc_neg"]:>9.3f}')
  for label, w in PRESETS:
    r = _eval_scores(lambda l, t, w=w: decision.score_logodds_multiscale(l, t, w), lex, ds)
    print(f'{label:<26}{r["nsmc"]["accuracy"]:>9.3f}{r["nikl"]["balanced_acc"]:>9.3f}'
          f'{r["nikl"]["roc_auc"]:>9.3f}{r["nikl"]["pr_auc_neg"]:>9.3f}')


if __name__ == '__main__':
  import sys
  main(sys.argv[1] if len(sys.argv) > 1 else None)
