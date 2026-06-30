"""Run a benchmark config against NSMC and/or NIKL and print the metric table.

A *config* is a plain dict (see :mod:`benchmarks.configs`):

    {'name': 'B0', 'lexicon': 'bundled', 'ngrams': [1,2,3], 'rule': 'logodds',
     'threshold': 0.0, 'negation': False, ...}

Usage::

    KOSAC_RUN_BENCH=1 python benchmarks/run_eval.py --config B0
    KOSAC_RUN_BENCH=1 python benchmarks/run_eval.py --all --json results/grid.json

The primary objective is **NSMC accuracy** (balanced, in-domain) and **NIKL
balanced-accuracy / PR-AUC(NEG)** (imbalanced, out-of-domain). NIKL accuracy is
never an objective (a constant-POS predictor scores ~97%).
"""
import argparse
import json
import os
import pickle

import numpy as np

from kosac.lexicon import PolarityLexicon, GenericLexicon

from . import datasets, decision
from .metrics import metrics, best_threshold, format_row

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, 'results', 'cache')

_TOKEN_CACHE = {}
_PLAIN_TOKENIZER = None


def _plain_tokenizer():
  global _PLAIN_TOKENIZER
  if _PLAIN_TOKENIZER is None:
    from kosac.tokenizers import KiwiTokenizer
    _PLAIN_TOKENIZER = KiwiTokenizer()
  return _PLAIN_TOKENIZER


def _spec_id(spec):
  if isinstance(spec, str):
    return spec
  if 'csv' in spec:
    return os.path.splitext(os.path.basename(spec['csv']))[0]
  if 'ensemble' in spec:
    return 'ens__' + '__'.join(_spec_id(s) for s in spec['ensemble'])
  raise ValueError(f'unknown lexicon spec: {spec!r}')


def build_lexicons(spec, ngrams, min_freq=0, lex_threshold=0.0):
  """Return ``(lexicons, weights)``. Single specs yield a 1-element list."""
  if isinstance(spec, dict) and 'ensemble' in spec:
    lexicons, weights = [], spec.get('weights')
    for sub in spec['ensemble']:
      lxs, _w = build_lexicons(sub, ngrams, min_freq, lex_threshold)
      lexicons.extend(lxs)
    return lexicons, weights

  if spec == 'bundled':
    lex = PolarityLexicon.load(ngrams=ngrams, min_freq=min_freq, threshold=lex_threshold)
  elif isinstance(spec, dict) and 'csv' in spec:
    lex = GenericLexicon(filepath=spec['csv'], ngrams=ngrams)
    if min_freq or lex_threshold:
      lex.set_lexicon(min_freq=min_freq, threshold=lex_threshold)
  else:
    raise ValueError(f'unknown lexicon spec: {spec!r}')
  return [lex], [1.0]


def get_token_lists(dataset_name, texts, tokenizer, tag):
  """Tokenize ``texts`` once per (dataset, tag), cached in memory and on disk."""
  key = (dataset_name, tag, len(texts))
  if key in _TOKEN_CACHE:
    return _TOKEN_CACHE[key]
  os.makedirs(CACHE_DIR, exist_ok=True)
  disk = os.path.join(CACHE_DIR, f'{dataset_name}__{tag}__{len(texts)}.pkl')
  if os.path.exists(disk):
    with open(disk, 'rb') as f:
      token_lists = pickle.load(f)
  else:
    print(f'  tokenizing {dataset_name} ({len(texts)} docs, tag={tag})...')
    token_lists = decision.tokenize_docs(tokenizer, texts)
    with open(disk, 'wb') as f:
      pickle.dump(token_lists, f, protocol=pickle.HIGHEST_PROTOCOL)
  _TOKEN_CACHE[key] = token_lists
  return token_lists


def evaluate(config, dataset):
  """Evaluate one config on one dataset dict ``{name, texts, pos}``."""
  spec = config['lexicon']
  ngrams = list(config.get('ngrams', [1, 2, 3]))
  rule = config.get('rule', 'logodds')
  lexicons, weights = build_lexicons(spec, ngrams, config.get('min_freq', 0),
                                     config.get('lex_threshold', 0.0))
  primary = lexicons[0]

  if config.get('align'):
    from kosac.tokenizers import KiwiTokenizer
    tokenizer = KiwiTokenizer.from_lexicon(primary)
    tag = 'align__' + _spec_id(spec)
  else:
    tokenizer = _plain_tokenizer()
    tag = 'plain'
  token_lists = get_token_lists(dataset['name'], dataset['texts'], tokenizer, tag)

  if rule in ('logodds', 'count_diff'):
    scorer = decision.score_logodds if rule == 'logodds' else decision.score_count_diff
    per = [scorer(lx, token_lists, ngrams) for lx in lexicons]
    scores, counts = decision.score_ensemble(per, weights) if len(per) > 1 else per[0]
  elif rule == 'softmax_margin':
    if len(lexicons) > 1:
      raise ValueError('softmax_margin rule does not support ensemble lexicons')
    analyzer = decision.make_analyzer(
        tokenizer, ngrams, negation=config.get('negation', False),
        intensifier=config.get('intensifier', False),
        smoothing=config.get('smoothing', True), window=config.get('window', 2),
        intensifier_factor=config.get('intensifier_factor', 2.0))
    scores, counts = decision.score_softmax_margin(analyzer, primary, token_lists,
                                                   dataset['texts'])
  else:
    raise ValueError(f'unknown rule {rule!r}; choose from {decision.RULES}')

  t = config.get('threshold', 0.0)
  row = metrics(f"{config['name']}@{dataset['name']}", scores, counts,
                dataset['pos'], threshold=t)
  row['config'] = config['name']
  row['dataset'] = dataset['name']
  row['rule'] = rule
  if config.get('oracle', True):
    ot, ov = best_threshold(scores, dataset['pos'], 'balanced_acc')
    row['oracle_threshold'] = ot
    row['oracle_balanced_acc'] = ov
  return row, scores, counts


def load_datasets(which):
  """Load requested datasets as ``{name, texts, pos, info}`` dicts."""
  out = []
  if 'nsmc' in which:
    texts, pos, info = datasets.load_nsmc_test()
    out.append({'name': 'nsmc', 'texts': texts, 'pos': pos, 'info': info})
  if 'nikl' in which:
    loaded = datasets.load_nikl()
    if loaded is None:
      print('NIKL skipped (data not present; set $KOSAC_NIKL_CSV or place '
            '.archive/nikl/nikl_docs_binary.csv)')
    else:
      texts, pos, info = loaded
      out.append({'name': 'nikl', 'texts': texts, 'pos': pos, 'info': info})
  return out


def run(configs, which=('nsmc', 'nikl')):
  ds = load_datasets(which)
  for d in ds:
    print(f"\n# {d['name']}: {len(d['texts'])} docs, POS-rate={d['pos'].mean():.1%} "
          f"({d['info'].get('file')}, sha={d['info']['sha256'][:12]})")
  rows = []
  for config in configs:
    print(f"\n=== {config['name']}: {config.get('desc', '')} ===")
    for d in ds:
      row, _s, _c = evaluate(config, d)
      print('  ' + format_row(row))
      if config.get('oracle', True):
        print(f"      oracle-threshold bal-acc={row['oracle_balanced_acc']:.1%} "
              f"(t={row['oracle_threshold']:.3f}, upper bound, tuned on test)")
      rows.append(row)
  return rows


def main():
  from .configs import CONFIGS
  ap = argparse.ArgumentParser()
  ap.add_argument('--config', action='append', help='config name(s) from configs.py')
  ap.add_argument('--all', action='store_true', help='run every config')
  ap.add_argument('--datasets', default='nsmc,nikl')
  ap.add_argument('--json', help='write the metric rows to this JSON path')
  args = ap.parse_args()

  if args.all:
    chosen = list(CONFIGS.values())
  elif args.config:
    chosen = [CONFIGS[name] for name in args.config]
  else:
    ap.error('pass --config NAME (repeatable) or --all')

  which = tuple(s.strip() for s in args.datasets.split(','))
  rows = run(chosen, which)
  if args.json:
    os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
    with open(args.json, 'w') as f:
      json.dump(rows, f, ensure_ascii=False, indent=2,
                default=lambda o: float(o) if isinstance(o, np.floating) else o)
    print(f'\nwrote {len(rows)} rows -> {args.json}')


if __name__ == '__main__':
  main()
