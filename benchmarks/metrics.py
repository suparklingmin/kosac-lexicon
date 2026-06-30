"""Binary-classification metrics for the polarity benchmark.

All metrics operate on a continuous per-document score (higher = more POS), a
per-document match count, the gold ``pos`` array (1 = POS, 0 = NEG), and a
decision threshold ``t`` (predict POS iff ``score > t``). The continuous score
lets the threshold-independent ranking metrics (ROC-AUC, PR-AUC) stay rule- and
threshold-agnostic, while a single tunable ``t`` handles class imbalance.

Ported from ``.archive/nikl/eval_lexicons.py::metrics`` with coverage measured
from an explicit match count (not ``score != 0``, which mislabels documents whose
matches happen to net to zero).
"""
import numpy as np

try:
  from sklearn.metrics import roc_auc_score, average_precision_score
except ImportError:  # pragma: no cover - sklearn ships in the [dev]/[sklearn] extra
  roc_auc_score = None
  average_precision_score = None


def _prf(pred, y, cls):
  """Precision / recall / F1 for predicting class ``cls`` against gold ``y``."""
  tp = int(((pred == cls) & (y == 1)).sum())
  fp = int(((pred == cls) & (y == 0)).sum())
  fn = int(((pred != cls) & (y == 1)).sum())
  p = tp / (tp + fp) if tp + fp else 0.0
  r = tp / (tp + fn) if tp + fn else 0.0
  f = 2 * p * r / (p + r) if p + r else 0.0
  return p, r, f


def metrics(name, scores, match_counts, pos, threshold=0.0):
  """Compute the full metric row for one (lexicon, rule, threshold) run.

  Parameters
  ----------
  name : str
      Display label for the run.
  scores : array-like of float
      Continuous per-doc POS score (higher = more POS).
  match_counts : array-like of int
      Number of lexicon matches per doc (for coverage). Pass ``None`` to skip.
  pos : array-like of int
      Gold labels, 1 = POS and 0 = NEG.
  threshold : float
      Predict POS iff ``score > threshold``.
  """
  scores = np.asarray(scores, dtype=float)
  pos = np.asarray(pos, dtype=int)
  neg = 1 - pos
  pred = (scores > threshold).astype(int)

  acc = float((pred == pos).mean())
  pp, pr, pf = _prf(pred, pos, 1)   # POS: predicted 1, gold indicator = pos
  np_, nr, nf = _prf(pred, neg, 0)  # NEG: predicted 0, gold indicator = neg
  bal = (pr + nr) / 2
  macro = (pf + nf) / 2

  if match_counts is None:
    cov = float('nan')
  else:
    cov = float((np.asarray(match_counts) > 0).mean())

  roc = _safe_auc(roc_auc_score, pos, scores)
  pr_neg = _safe_auc(average_precision_score, neg, -scores)

  return dict(name=name, threshold=float(threshold), n=int(len(pos)),
              coverage=cov, accuracy=acc, balanced_acc=bal, macro_f1=macro,
              pos_p=pp, pos_r=pr, pos_f1=pf, neg_p=np_, neg_r=nr, neg_f1=nf,
              neg_recall=nr, neg_recall_n=int(((pred == 0) & (neg == 1)).sum()),
              neg_total=int(neg.sum()), roc_auc=roc, pr_auc_neg=pr_neg,
              neg_prevalence=float(neg.mean()))


def _safe_auc(fn, y, scores):
  """Run an sklearn ranking metric, returning NaN if it is undefined here."""
  if fn is None:
    return float('nan')
  y = np.asarray(y)
  if y.min() == y.max():  # only one class present -> AUC undefined
    return float('nan')
  try:
    return float(fn(y, scores))
  except ValueError:
    return float('nan')


def best_threshold(scores, pos, objective='balanced_acc'):
  """Threshold that maximizes balanced accuracy ON THIS SET (oracle upper bound).

  Returns ``(threshold, balanced_acc)``. Vectorized via cumulative class counts
  over the sorted scores (O(n log n)), only at true value boundaries so the
  result is achievable by a single ``score > t`` rule. Use only for the
  clearly-labeled "oracle-threshold" row — never as a headline number, since it
  is tuned on the set it is evaluated on.
  """
  if objective != 'balanced_acc':  # pragma: no cover - only metric we tune on
    raise ValueError('best_threshold only supports balanced_acc')
  scores = np.asarray(scores, dtype=float)
  pos = np.asarray(pos, dtype=int)
  P, N = int(pos.sum()), int((1 - pos).sum())
  if P == 0 or N == 0:
    return 0.0, 0.0

  order = np.argsort(scores, kind='mergesort')
  s, y = scores[order], pos[order]
  # Split k predicts POS for the suffix s[k:] (largest scores), NEG for s[:k].
  cum_pos = np.concatenate(([0], np.cumsum(y)))
  cum_neg = np.concatenate(([0], np.cumsum(1 - y)))
  k = np.arange(len(s) + 1)
  tpr = (P - cum_pos[k]) / P          # POS recall
  tnr = cum_neg[k] / N                # NEG recall
  bal = (tpr + tnr) / 2
  # Only splits at true value boundaries are achievable by one threshold.
  valid = np.ones(len(s) + 1, dtype=bool)
  valid[1:len(s)] = s[1:] != s[:-1]
  bal = np.where(valid, bal, -1.0)
  best = int(bal.argmax())

  if best == 0:
    t = s[0] - 1.0
  elif best == len(s):
    t = s[-1] + 1.0
  else:
    t = (s[best - 1] + s[best]) / 2.0
  return float(t), float(bal[best])


def format_row(row):
  """One-line human summary of a metric dict from :func:`metrics`."""
  return (f'{row["name"]:<28} cov={row["coverage"]:.1%} acc={row["accuracy"]:.1%} '
          f'bal={row["balanced_acc"]:.1%} mF1={row["macro_f1"]:.3f} '
          f'NEGr={row["neg_recall"]:.1%} ROC={row["roc_auc"]:.3f} '
          f'PR(NEG)={row["pr_auc_neg"]:.3f}')
