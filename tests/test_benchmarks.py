"""Smoke tests for the benchmark harness math (no network, no Kiwi).

Guards the decision rules, the tunable threshold, and the metric formulas with
tiny hand-checked fixtures so the harness logic is regression-tested in CI while
the full NSMC/NIKL run stays opt-in (``KOSAC_RUN_BENCH=1``).
"""
import os
import sys

import numpy as np
import pytest

# benchmarks/ is a repo-root package, not installed; make it importable.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks import decision
from benchmarks.metrics import metrics, best_threshold


# --- metric math ----------------------------------------------------------
def test_metrics_perfect_separation():
    scores = np.array([2.0, -1.0, 0.5, -3.0])
    pos = np.array([1, 0, 1, 0])
    row = metrics('perfect', scores, [1, 1, 1, 1], pos, threshold=0.0)
    assert row['accuracy'] == 1.0
    assert row['balanced_acc'] == 1.0
    assert row['macro_f1'] == 1.0
    assert row['roc_auc'] == 1.0
    assert row['coverage'] == 1.0


def test_metrics_known_values():
    # pred = score>0 -> [1,1,0,0]; gold pos = [1,0,1,0]
    scores = np.array([2.0, 1.0, -0.5, -3.0])
    pos = np.array([1, 0, 1, 0])
    row = metrics('known', scores, [2, 1, 1, 0], pos, threshold=0.0)
    assert row['accuracy'] == 0.5
    assert row['pos_p'] == pytest.approx(0.5)
    assert row['pos_r'] == pytest.approx(0.5)
    assert row['neg_p'] == pytest.approx(0.5)
    assert row['neg_r'] == pytest.approx(0.5)
    assert row['balanced_acc'] == pytest.approx(0.5)
    assert row['coverage'] == pytest.approx(0.75)  # 3 of 4 docs matched


def test_threshold_shifts_predictions():
    scores = np.array([2.0, 1.0, -0.5, -3.0])
    pos = np.array([1, 1, 0, 0])
    # raising the threshold to 1.5 predicts POS only for doc0 -> [1,0,0,0]
    row = metrics('thr', scores, None, pos, threshold=1.5)
    assert row['accuracy'] == 0.75  # doc1 now wrong, rest right


def test_best_threshold_is_oracle_upper_bound():
    scores = np.array([2.0, 1.0, -0.5, -3.0])
    pos = np.array([1, 0, 1, 0])
    base = metrics('b', scores, None, pos, threshold=0.0)['balanced_acc']
    _t, val = best_threshold(scores, pos, 'balanced_acc')
    assert val >= base  # tuned on the same set: never worse than t=0


def test_safe_auc_single_class_is_nan():
    scores = np.array([1.0, 2.0, 3.0])
    pos = np.array([1, 1, 1])  # only one class -> AUC undefined
    row = metrics('one', scores, None, pos)
    assert np.isnan(row['roc_auc'])


# --- decision rules (mini_lexicon fixture from conftest, ngrams=[1]) -------
def _toks(*surfaces):
    return [(s, i, i + 1) for i, s in enumerate(surfaces)]


def test_logodds_rule_signs(mini_lexicon):
    # 좋/VA: POS=5,NEG=0 -> +;  힘/NNG: POS=2,NEG=7 -> -;  unknown -> 0
    docs = [_toks('좋/VA'), _toks('힘/NNG'), _toks('없는단어/NNG')]
    scores, counts = decision.score_logodds(mini_lexicon, docs, [1])
    assert scores[0] > 0
    assert scores[1] < 0
    assert scores[2] == 0.0
    assert list(counts) == [1, 1, 0]


def test_count_diff_rule(mini_lexicon):
    docs = [_toks('좋/VA', '힘/NNG'), _toks('힘/NNG'), _toks('좋/VA')]
    scores, counts = decision.score_count_diff(mini_lexicon, docs, [1])
    assert scores[0] == 0   # one POS-dominant + one NEG-dominant
    assert scores[1] == -1  # 힘/NNG is NEG-dominant
    assert scores[2] == 1   # 좋/VA is POS-dominant
    assert list(counts) == [2, 1, 1]


def test_ensemble_sums_scores_and_counts(mini_lexicon):
    docs = [_toks('좋/VA')]
    one = decision.score_logodds(mini_lexicon, docs, [1])
    combined = decision.score_ensemble([one, one], weights=[1.0, 1.0])
    assert combined[0][0] == pytest.approx(2 * one[0][0])
    assert combined[1][0] == 2 * one[1][0]


def test_multiscale_weights_scale_and_gate(mini_lexicon):
    # mini_lexicon is unigram-only, so multiscale weight on length 1 is the knob.
    docs = [_toks('좋/VA'), _toks('힘/NNG')]
    s1, c1 = decision.score_logodds_multiscale(mini_lexicon, docs, {1: 1, 2: 1, 3: 1})
    assert s1[0] > 0 and s1[1] < 0          # 좋 POS, 힘 NEG
    s2, _ = decision.score_logodds_multiscale(mini_lexicon, docs, {1: 2})
    assert s2[0] == pytest.approx(2 * s1[0])  # weight scales the contribution
    s0, c0 = decision.score_logodds_multiscale(mini_lexicon, docs, {1: 0})
    assert s0[0] == 0.0 and list(c0) == [0, 0]  # zero weight gates the length out


def test_softmax_margin_via_analyzer(mini_lexicon):
    # Uses the analyzer's _score with a plain whitespace tokenizer (no Kiwi);
    # bundled polarity data is loaded internally but the mini lexicon is scored.
    from kosac.tokenizers import Tokenizer
    analyzer = decision.make_analyzer(Tokenizer(), ngrams=[1])
    docs = [_toks('좋/VA'), _toks('힘/NNG')]
    texts = ['좋다', '힘들다']
    scores, counts = decision.score_softmax_margin(analyzer, mini_lexicon, docs, texts)
    assert scores[0] > 0   # 좋/VA leans POS
    assert scores[1] < 0   # 힘/NNG leans NEG
    assert list(counts) == [1, 1]
