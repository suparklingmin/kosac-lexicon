"""Named benchmark configs: baselines (B*) and tuning experiments (A*, B-tier, C*).

Each config is a dict consumed by :func:`benchmarks.run_eval.evaluate`. Keep them
declarative and reproducible: the config name is the unit recorded in result JSON
and in ``benchmarks/README.md``.

Lexicon specs:
  'bundled'                    frozen 2016 KOSAC polarity (PolarityLexicon.load)
  {'csv': PATH}                a CSV lexicon (learned / derived / blended)
  {'ensemble': [s1, s2],       score-level ensemble (logodds/count_diff only),
   'weights': [w1, w2]}        weights default to equal

Knobs: ngrams, rule ('logodds'|'count_diff'|'softmax_margin'), threshold,
min_freq, lex_threshold (max.prop), and (softmax_margin only) negation,
intensifier, intensifier_factor, smoothing, window; align (Kiwi user-dict seed).
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
ARCHIVE = os.path.join(REPO, '.archive', 'nikl')

# External lexicons used as references / Tier-C inputs (not shipped).
NSMC_ALL_CSV = os.path.join(ARCHIVE, 'nsmc-lexicon-all.csv')

# Built artifacts (created by build_derived.py / build_blend.py / Tier-C scripts).
DERIVED_DIR = os.path.join(HERE, 'results', 'lexicons')


def _derived(name):
  return os.path.join(DERIVED_DIR, name)


NG3 = [1, 2, 3]

# ---- Phase 0b: baselines -------------------------------------------------
BASELINES = [
    dict(name='B0', desc='bundled KOSAC, logodds, t=0 (reference)',
         lexicon='bundled', ngrams=NG3, rule='logodds', threshold=0.0),
    dict(name='B1', desc='bundled KOSAC, softmax_margin (analyzer path)',
         lexicon='bundled', ngrams=NG3, rule='softmax_margin', threshold=0.0),
    dict(name='B2', desc='bundled KOSAC, unigrams only, logodds',
         lexicon='bundled', ngrams=[1], rule='logodds', threshold=0.0),
    dict(name='B3', desc='NSMC-trained (all tags), logodds, t=0',
         lexicon={'csv': NSMC_ALL_CSV}, ngrams=NG3, rule='logodds', threshold=0.0),
    dict(name='B4', desc='bundled KOSAC, count_diff (cheap floor)',
         lexicon='bundled', ngrams=NG3, rule='count_diff', threshold=0.0),
]

# ---- Phase 1, Tier A: analyzer / parameter tuning (no new data) ----------
TIER_A = [
    # A2 n-gram set
    dict(name='A2-12', desc='bundled, ngrams (1,2), logodds',
         lexicon='bundled', ngrams=[1, 2], rule='logodds'),
    # A3 min_freq / max.prop threshold sweep
    dict(name='A3-mf5', desc='bundled, min_freq>=5, logodds',
         lexicon='bundled', ngrams=NG3, rule='logodds', min_freq=5),
    dict(name='A3-mf2', desc='bundled, min_freq>=2, logodds',
         lexicon='bundled', ngrams=NG3, rule='logodds', min_freq=2),
    dict(name='A3-th7', desc='bundled, max.prop>0.7, logodds',
         lexicon='bundled', ngrams=NG3, rule='logodds', lex_threshold=0.7),
    # A4 negation (softmax_margin so composition is live)
    dict(name='A4-neg', desc='bundled, softmax_margin + negation',
         lexicon='bundled', ngrams=NG3, rule='softmax_margin', negation=True),
    # A5 intensifier
    dict(name='A5-int', desc='bundled, softmax_margin + intensifier',
         lexicon='bundled', ngrams=NG3, rule='softmax_margin', intensifier=True),
    dict(name='A45', desc='bundled, softmax_margin + negation + intensifier',
         lexicon='bundled', ngrams=NG3, rule='softmax_margin', negation=True,
         intensifier=True),
    # A6 align (Kiwi user-dict seeded with lexicon unigrams)
    dict(name='A6-align', desc='bundled, logodds, align tokenizer',
         lexicon='bundled', ngrams=NG3, rule='logodds', align=True),
]

# Threshold sweep (A1) is generated programmatically against a chosen base.
def threshold_sweep(base_name, base, thresholds):
  out = []
  for t in thresholds:
    out.append(dict(base, name=f'{base_name}-t{t:+.2f}',
                    desc=f"{base.get('desc','')} @t={t:+.2f}", threshold=t))
  return out


# ---- Phase 2, Tier B: derived / cleaned lexicons (build_derived.py) -------
TIER_B = [
    dict(name='Bd-nowild', desc='bundled minus dead wildcard entries, logodds',
         lexicon={'csv': _derived('polarity-nowild.csv')}, ngrams=NG3, rule='logodds'),
    dict(name='Bd-clean', desc='nowild + ambiguous function words pruned, logodds',
         lexicon={'csv': _derived('polarity-clean.csv')}, ngrams=NG3, rule='logodds'),
]

# ---- Phase 3, Tier C: corpus-learned (build_nsmc.py) + blends ------------
NSMC_TRAIN_ALL = _derived('nsmc-train-all.csv')
NSMC_TRAIN_CONTENT = _derived('nsmc-train-content.csv')
TIER_C = [
    dict(name='C1-all', desc='NSMC-train (all tags), logodds (leakage-safe)',
         lexicon={'csv': NSMC_TRAIN_ALL}, ngrams=NG3, rule='logodds'),
    dict(name='C1-content', desc='NSMC-train (content tags), logodds',
         lexicon={'csv': NSMC_TRAIN_CONTENT}, ngrams=NG3, rule='logodds'),
    # C2(i) union blend: bundled + NSMC-train (built by build_blend.py)
    dict(name='C2-blend', desc='union blend bundled+NSMC-train(all), logodds',
         lexicon={'csv': _derived('blend-nsmc-train-all.csv')}, ngrams=NG3, rule='logodds'),
    # C2(ii) score-level ensemble (equal weight): cleaner, no build step
    dict(name='C2-ens', desc='score-ensemble bundled+NSMC-train(all), logodds',
         lexicon={'ensemble': ['bundled', {'csv': NSMC_TRAIN_ALL}], 'weights': [1.0, 1.0]},
         ngrams=NG3, rule='logodds'),
    # NSMC-train scored by count_diff (discrete vote — robust on imbalanced NIKL)
    dict(name='C1-all-cd', desc='NSMC-train (all), count_diff',
         lexicon={'csv': NSMC_TRAIN_ALL}, ngrams=NG3, rule='count_diff'),
]

# ---- Follow-up round: function-only unigram removal + blend-ratio sweep ----
FOLLOWUP = [
    # (1) drop ALL function-only unigrams from the bundled lexicon
    dict(name='Bd-nofunc', desc='bundled minus function-only unigrams, logodds',
         lexicon={'csv': _derived('polarity-nofunc.csv')}, ngrams=NG3, rule='logodds'),
    dict(name='Bd-nofunc-cd', desc='bundled minus function-only unigrams, count_diff',
         lexicon={'csv': _derived('polarity-nofunc.csv')}, ngrams=NG3, rule='count_diff'),

    # (2a) blend ratio in SCORE space: weights = [bundled, nsmc], nsmc fixed at 1
    dict(name='E-wb0.0', desc='score-ensemble wb=0 (pure NSMC sanity)',
         lexicon={'ensemble': ['bundled', {'csv': NSMC_TRAIN_ALL}], 'weights': [0.0, 1.0]},
         ngrams=NG3, rule='logodds'),
    dict(name='E-wb0.25', desc='score-ensemble bundled:nsmc = 0.25:1',
         lexicon={'ensemble': ['bundled', {'csv': NSMC_TRAIN_ALL}], 'weights': [0.25, 1.0]},
         ngrams=NG3, rule='logodds'),
    dict(name='E-wb0.5', desc='score-ensemble bundled:nsmc = 0.5:1',
         lexicon={'ensemble': ['bundled', {'csv': NSMC_TRAIN_ALL}], 'weights': [0.5, 1.0]},
         ngrams=NG3, rule='logodds'),
    dict(name='E-wb2', desc='score-ensemble bundled:nsmc = 2:1',
         lexicon={'ensemble': ['bundled', {'csv': NSMC_TRAIN_ALL}], 'weights': [2.0, 1.0]},
         ngrams=NG3, rule='logodds'),

    # (2b) blend ratio in COUNT space: bundled counts x alpha, then union
    dict(name='U-a5', desc='union blend, bundled x5',
         lexicon={'csv': _derived('blend-nsmc-train-all-a5.csv')}, ngrams=NG3, rule='logodds'),
    dict(name='U-a20', desc='union blend, bundled x20',
         lexicon={'csv': _derived('blend-frozen-a20.csv')}, ngrams=NG3, rule='logodds'),
    dict(name='U-a100', desc='union blend, bundled x100',
         lexicon={'csv': _derived('blend-nsmc-train-all-a100.csv')}, ngrams=NG3, rule='logodds'),
    # (1)+(2b): nofunc-cleaned bundled x20 union NSMC
    dict(name='U-nofunc-a20', desc='union blend, nofunc-bundled x20',
         lexicon={'csv': _derived('blend-nofunc-a20.csv')}, ngrams=NG3, rule='logodds'),
]

# Registry of all individually-runnable configs.
CONFIGS = {c['name']: c for c in (BASELINES + TIER_A + TIER_B + TIER_C + FOLLOWUP)}
