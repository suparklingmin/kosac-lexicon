"""Dataset loaders for the polarity benchmark.

Two corpora, with a strict train/test discipline (see ``benchmarks/README.md``):

* **NSMC** (in-domain, balanced binary, CC0) — Naver Sentiment Movie Corpus.
  Downloaded from github.com/e9t/nsmc and cached in the system temp dir. The
  official split is honored: a lexicon is learned ONLY from ``ratings_train.txt``
  and evaluated ONLY on ``ratings_test.txt``. SHA-256 + row counts are recorded
  so a run is verifiable.
* **NIKL** (out-of-domain holdout, ~97% POS) — document-level binary labels built
  from the NIKL sentiment-analysis corpus. The CSV is license-restricted and
  cannot ship; it is read from ``$KOSAC_NIKL_CSV`` or the local ``.archive`` copy,
  and loading degrades gracefully (returns ``None``) when absent. NIKL is NEVER
  trained on — it only ever appears on the evaluation side.
"""
import hashlib
import os
import tempfile
import urllib.request

import numpy as np
import pandas as pd

from kosac.corpora import Corpus

NSMC_BASE = 'https://raw.githubusercontent.com/e9t/nsmc/master'
NSMC_FILES = {'train': 'ratings_train.txt', 'test': 'ratings_test.txt'}

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
NIKL_DEFAULT = os.path.join(REPO, '.archive', 'nikl', 'nikl_docs_binary.csv')


def _cache_dir():
  path = os.path.join(tempfile.gettempdir(), 'kosac_nsmc')
  os.makedirs(path, exist_ok=True)
  return path


def sha256(path):
  h = hashlib.sha256()
  with open(path, 'rb') as f:
    for chunk in iter(lambda: f.read(1 << 20), b''):
      h.update(chunk)
  return h.hexdigest()


def _fetch_nsmc(split):
  fname = NSMC_FILES[split]
  path = os.path.join(_cache_dir(), fname)
  if not os.path.exists(path):
    url = f'{NSMC_BASE}/{fname}'
    print(f'downloading NSMC {split} -> {path}')
    urllib.request.urlretrieve(url, path)
  return path


def _read_nsmc(split):
  """Return the cleaned NSMC dataframe with a ``POS``/``NEG`` ``label`` column."""
  path = _fetch_nsmc(split)
  df = pd.read_csv(path, sep='\t', quoting=3, keep_default_na=False)
  df['document'] = df['document'].astype(str).str.strip()
  df = df[df['document'] != '']
  df = df.rename(columns={'document': 'text'})
  df['label'] = df['label'].map({1: 'POS', 0: 'NEG'})
  return df[['text', 'label']].reset_index(drop=True), path


def _corpus_from_df(df):
  """Wrap a ``text,label`` dataframe in a Corpus without going through a file."""
  corpus = Corpus.__new__(Corpus)
  corpus.df = df.reset_index(drop=True).copy()
  corpus.labels = corpus.df['label'].unique().tolist()
  return corpus


def load_nsmc_train():
  """NSMC train split as a :class:`~kosac.corpora.Corpus` (for learning)."""
  df, path = _read_nsmc('train')
  return _corpus_from_df(df), {'file': os.path.basename(path),
                               'sha256': sha256(path), 'rows': len(df)}


def load_nsmc_test():
  """NSMC test split as ``(texts, pos)`` for evaluation (never trained on)."""
  df, path = _read_nsmc('test')
  pos = (df['label'] == 'POS').astype(int).to_numpy()
  return df['text'].tolist(), pos, {'file': os.path.basename(path),
                                    'sha256': sha256(path), 'rows': len(df)}


def split_train_val(corpus, val_frac=0.1, seed=0):
  """Deterministic train/val split of a corpus for THRESHOLD tuning only.

  The val slice never overlaps the reported NSMC test set (a separate file).
  """
  df = corpus.df.reset_index(drop=True)
  rng = np.random.default_rng(seed)
  idx = rng.permutation(len(df))
  n_val = int(len(df) * val_frac)
  val_idx, train_idx = idx[:n_val], idx[n_val:]
  train = _corpus_from_df(df.iloc[train_idx])
  val_df = df.iloc[val_idx].reset_index(drop=True)
  val_pos = (val_df['label'] == 'POS').astype(int).to_numpy()
  return train, val_df['text'].tolist(), val_pos


def load_nikl(path=None):
  """NIKL doc-binary holdout as ``(texts, pos, info)``; ``None`` if unavailable."""
  path = path or os.environ.get('KOSAC_NIKL_CSV') or NIKL_DEFAULT
  if not os.path.exists(path):
    return None
  df = pd.read_csv(path, names=('text', 'label'), keep_default_na=False)
  pos = (df['label'] == 'POS').astype(int).to_numpy()
  info = {'file': os.path.basename(path), 'sha256': sha256(path), 'rows': len(df),
          'pos_rate': float(pos.mean())}
  return df['text'].tolist(), pos, info
