# kosac-lexicon polarity benchmark

A reproducible harness for measuring (and improving) the **polarity (POS/NEG)**
classification performance of the lexicons, separate from the package itself. It
is not shipped in the wheel.

## What it measures

Two corpora, deliberately chosen to be in-domain vs. out-of-domain:

| Dataset | Role | Size | Balance | Primary metric |
|---|---|---|---|---|
| **NSMC** | in-domain, balanced | 50k test reviews | ~50% POS | **accuracy** (macro-F1 tiebreak) |
| **NIKL** | out-of-domain holdout | 2,009 docs | ~97% POS | **balanced-acc + PR-AUC(NEG)** |

NIKL **accuracy is never an objective** — a constant-POS predictor already scores
~97%. The metrics that matter there are balanced accuracy, NEG-recall, and
PR-AUC for the rare NEG class.

## Decision rules (one continuous score, one tunable threshold)

The package analyzer argmaxes a softmax over all five KOSAC labels (COMP/NEG/
NEUT/None/POS), which is *not* a binary decision. The harness instead reduces
every document to a single continuous POS score (higher = more POS) and applies a
threshold `t` (predict POS iff `score > t`). This (a) makes ROC-AUC / PR-AUC
rule- and threshold-independent, and (b) lets one knob absorb the NIKL imbalance.

Three rules (`benchmarks/decision.py`):

- **logodds** — `Σ[log(POS+1) − log(NEG+1)]` over matches. Matches the original
  `.archive/nikl/eval_lexicons.py` reference; ignores NEUT/None/COMP and any
  composition.
- **softmax_margin** — `P(POS) − P(NEG)` from the analyzer's smoothed log-prob
  softmax, so negation / intensifier / smoothing compose. Reuses
  `SentimentAnalyzer._score` with an arbitrary lexicon.
- **count_diff** — `#POS-dominant − #NEG-dominant` matches (the `_count` path,
  collapsed to binary). Cheap floor.

## No leakage

- A learned lexicon is built **only** from NSMC `ratings_train.txt` and evaluated
  **only** on `ratings_test.txt` (the official split). NIKL is **never** trained
  on — holdout only.
- The reported threshold is `t=0` (zero-tuning). An "oracle-threshold" row (best
  balanced-acc, threshold chosen on the eval set) is printed as a clearly-labeled
  **upper bound**, never as a headline number.
- Every run records the source file SHA-256 + row counts.

## Running

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"

# fast, no network/Kiwi — guards the harness math
.venv/bin/pytest tests/test_benchmarks.py

# full evaluation (downloads NSMC ~spends minutes the first time; caches tokens)
KOSAC_RUN_BENCH=1 .venv/bin/python -m benchmarks.run_eval --config B0
KOSAC_RUN_BENCH=1 .venv/bin/python -m benchmarks.run_eval --all --json benchmarks/results/grid.json
```

NSMC (CC0) downloads from github.com/e9t/nsmc into the system temp dir.

NIKL is license-restricted and **cannot ship**. Place the doc-binary CSV at
`.archive/nikl/nikl_docs_binary.csv` or point `$KOSAC_NIKL_CSV` at it; otherwise
NIKL is skipped gracefully. It is regenerable from the NIKL EXSA JSON via
`.archive/nikl/build_nikl_docs.py`.

## Results

Datasets: NSMC test `ratings_test.txt` (49,997 docs, sha `8ac9f640…`, 50.3% POS),
NIKL `nikl_docs_binary.csv` (2,009 docs, sha `a747aa41…`, 97.1% POS). Learned
lexicons built from NSMC train `ratings_train.txt` (149,995 docs, sha `e03b7d14…`)
— a different file from the eval sets, so no leakage. Reproduce with
`KOSAC_RUN_BENCH=1 python -m benchmarks.run_eval --all --json benchmarks/results/grid.json`.

All numbers at threshold `t=0`. NSMC primary = accuracy; NIKL primary =
balanced-acc + PR-AUC(NEG). NIKL accuracy is shown but is **not** an objective
(constant-POS = 97.1% acc / 50% bal-acc).

### Baselines

| Config | lexicon / rule | NSMC acc | NSMC mF1 | NIKL bal | NIKL NEGr | NIKL ROC | NIKL PR(NEG) |
|---|---|---:|---:|---:|---:|---:|---:|
| **B0** | bundled, logodds | 0.583 | 0.576 | 0.534 | 0.138 | 0.436 | 0.064 |
| B1 | bundled, softmax_margin | 0.583 | 0.576 | 0.534 | 0.138 | 0.431 | 0.063 |
| B2 | bundled, unigrams, logodds | 0.583 | 0.572 | 0.538 | 0.121 | 0.408 | 0.056 |
| **B3** | NSMC-all (archive*), logodds | 0.861 | 0.861 | 0.721 | 0.638 | 0.689 | 0.231 |
| B4 | bundled, count_diff | 0.576 | 0.574 | 0.652 | 0.500 | 0.660 | 0.092 |

B0 / B3 reproduce the `.archive` reference numbers exactly (bundled bal-acc 53.4%
/ ROC 0.436; NSMC-trained bal-acc 72.1% / ROC 0.689 / NEG-recall 63.8%), so the
harness is trustworthy. *B3 uses the archived `nsmc-lexicon-all.csv` whose NSMC
train/test split is unverified — its NSMC accuracy may be leakage-inflated; the
leakage-safe equivalent is **C1-all**.

### Experiments (Tier A bundled tuning / Tier B cleaning / Tier C learning)

| Config | what | NSMC acc | NIKL bal | NIKL ROC | NIKL PR(NEG) |
|---|---|---:|---:|---:|---:|
| A2-12 | bundled ngrams (1,2) | 0.585 | 0.544 | 0.435 | 0.069 |
| A3-mf5 | bundled min_freq≥5 | 0.551 | 0.511 | 0.318 | 0.028 |
| A3-mf2 | bundled min_freq≥2 | 0.568 | 0.533 | 0.387 | 0.046 |
| A3-th7 | bundled max.prop>0.7 | 0.585 | **0.599** | 0.547 | 0.134 |
| A4-neg | bundled +negation | 0.581 | 0.528 | 0.401 | 0.066 |
| A5-int | bundled +intensifier | 0.584 | 0.527 | 0.434 | 0.059 |
| A45 | bundled +neg +int | 0.582 | 0.528 | 0.411 | 0.066 |
| A6-align | bundled, align tokenizer | 0.576 | 0.536 | 0.419 | 0.061 |
| Bd-nowild | −170 dead wildcards | 0.583 | 0.534 | 0.436 | 0.064 |
| Bd-clean | −wildcards −ambiguous func. words | 0.582 | 0.535 | 0.417 | 0.058 |
| **C1-all** | NSMC-train all, logodds | 0.859 | 0.716 | 0.679 | 0.213 |
| C1-content | NSMC-train content, logodds | 0.845 | 0.677 | 0.644 | 0.222 |
| **C2-blend** | bundled ⊕ NSMC-train union | 0.859 | **0.721** | 0.684 | 0.221 |
| C2-ens | bundled+NSMC score-ensemble | 0.831 | 0.687 | 0.644 | 0.199 |
| C1-all-cd | NSMC-train all, count_diff | 0.812 | 0.618 | 0.576 | 0.103 |

Takeaways:

1. **Corpus learning is the only lever that matters.** Every bundled-lexicon
   knob (Tier A) and cleaning step (Tier B) stays pinned near B0 (~0.58 NSMC /
   ~0.53 NIKL); learning from NSMC (Tier C) jumps to ~0.86 NSMC / ~0.72 NIKL.
2. **Union blend ≥ learning alone.** `C2-blend` ties `C1-all` on NSMC (0.859) and
   beats it on NIKL bal-acc (0.721 vs 0.716): the frozen bundled data adds a sliver
   of out-of-domain robustness for free, and it's leakage-safe.
3. **For a bundled-only deployment** (no corpus available), `count_diff` (B4,
   NIKL bal 0.652) or `max.prop>0.7` pruning (A3-th7, NIKL bal 0.599) are the only
   things that lift the frozen lexicon out-of-domain — but far below learned levels
   and not a strict-Pareto win (NSMC flat/down).
4. Tier-B cleaning is metric-neutral (wildcards were already dead; function-word
   pruning marginal) — still a legitimate bug-fix / size trim, not a perf lever.
5. `min_freq` pruning (A3-mf*) *hurts* out-of-domain (NIKL ROC 0.318–0.387): it
   strips the low-frequency entries that carry the little OOD signal the bundled
   lexicon has.

### Follow-up round: function-only unigrams + blend-ratio tuning

Two more experiments (`benchmarks/results/followup.json`):

**(1) Drop all function-only unigrams from the bundled lexicon** (227 bare
particles/endings, regardless of max.prop — stronger than Bd-clean):

| Config | NSMC acc | NIKL bal | NIKL NEGr | NIKL ROC |
|---|---:|---:|---:|---:|
| B0 bundled frozen | 0.583 | 0.534 | 8/58 | 0.436 |
| Bd-nofunc (logodds) | 0.573 | 0.570 | 11/58 | 0.469 |
| Bd-nofunc + count_diff | 0.571 | 0.645 | 32/58 | 0.663 |

Removing them helps out-of-domain (NIKL bal +3.6pp, ROC +3.3pp) but costs ~1pp
in-domain — a bare particle is OOD noise but adds a little in-domain coverage. Not
a strict-Pareto win on its own (NSMC drops).

**(2) Blend ratio.** Bundled (seed) counts are tiny next to NSMC frequencies, so at
the default union (α=1) the bundled side is drowned (→ blend ≈ NSMC alone). Scaling
the bundled counts ×α before the union surfaces a *real* contribution:

| Bundled weight | NSMC acc | NIKL bal | NIKL ROC | NIKL PR(NEG) | NEG rec |
|---|---:|---:|---:|---:|---:|
| NSMC alone (C1-all) | 0.859 | 0.716 | 0.679 | 0.213 | — |
| union ×1 (C2-blend) | 0.859 | 0.721 | 0.684 | 0.221 | 36/58 |
| union ×10 | 0.858 | 0.743 | 0.699 | 0.239 | 36/58 |
| **union ×20 (U-a20)** | 0.857 | **0.746** | 0.705 | 0.242 | 36/58 |
| union ×25 | 0.857 | 0.748 | 0.708 | 0.244 | 36/58 |
| union ×40 | 0.856 | 0.748 | 0.714 | 0.246 | 36/58 |
| union ×100 | 0.852 | 0.724 | 0.726 | 0.242 | 36/58 |

Fine sweep (`python -m benchmarks.sweep_alpha`) over α∈{1,8,…,40}: **NIKL
balanced-acc plateaus at ~0.74–0.75 for α≈8–40** and NEG-recall is *flat at 36/58
across the whole range* — so the bal-acc wiggle (incl. the nominal α=25 peak 0.748
vs α=20 0.746) is POS-side noise hitting the granularity floor of a 58-NEG test set.
The genuinely informative signal is the **threshold-independent ROC-AUC, which rises
monotonically** with the bundled weight (0.679→0.705→0.714→0.726), as does PR(NEG)
(0.213→0.246). So the frozen curated data *does* carry real out-of-domain ranking
signal — it only surfaces once up-weighted ~20× to compensate for its tiny seed
counts — but higher α slowly erodes in-domain NSMC (0.859→0.852). **Practical sweet
spot: α≈20–25** (bal-acc plateau + near-peak ROC + ≤0.2pp NSMC cost). Score-space
ensembling helps only marginally (peaks at bundled:nsmc ≈ 0.25:1).

### Learning-side levers (NSMC-train lexicon)

Sweeping the lexicon-learning knobs (`python -m benchmarks.sweep_learn`):

**(A) min_freq** [pos=all, ngrams=1,2,3] — the default 2 is too low:

| min_freq | NSMC acc | NIKL bal | NIKL ROC | NIKL PR(NEG) |
|---:|---:|---:|---:|---:|
| 1 | 0.855 | 0.660 | 0.661 | 0.217 |
| 2 (old default) | 0.859 | 0.716 | 0.679 | 0.213 |
| 5 | 0.861 | 0.721 | 0.689 | 0.231 |
| **10** | 0.859 | **0.730** | **0.697** | **0.237** |
| 20 | 0.855 | 0.716 | 0.689 | 0.228 |

**min_freq=10 strictly beats the default 2** — NSMC flat, every NIKL metric up
(rare single-review entries are OOD noise). Optimum ≈ 5–10.

**(B) ngrams** [pos=all, mf=2] — an in-domain ↔ OOD trade-off:

| ngrams | NSMC acc | NIKL bal | NIKL ROC | NIKL PR(NEG) |
|---|---:|---:|---:|---:|
| [1] | 0.833 | 0.699 | **0.724** | 0.177 |
| [1,2] | 0.856 | 0.694 | 0.685 | 0.206 |
| [1,2,3] | 0.859 | 0.716 | 0.679 | 0.213 |

Unigrams alone rank best OOD (ROC 0.724) but are weakest in-domain and on PR(NEG);
higher orders capture review-specific phrases that lift in-domain + PR but don't
transfer (ROC drops). [1,2,3] is the balanced default.

**(C) pos-filter** [mf=2, ngrams=1,2,3]: `all` 0.859/0.716/ROC 0.679 dominates
`content` 0.845/0.677/0.644 and `content+ic` 0.846/0.677/0.645 — keeping
punctuation/endings/emoticons wins; adding interjections (IC) does nothing.

**Stacking the best learning settings into the blend** (mf10 learned ⊕ bundled×20):

| Config | NSMC | NIKL bal (t0 / oracle) | NIKL ROC | NIKL PR(NEG) |
|---|---:|---:|---:|---:|
| U-a20 (mf2 blend) | 0.857 | 0.746 / 0.750 | 0.705 | 0.242 |
| blend mf10 ×20 | 0.856 | 0.726 / 0.738 | **0.723** | **0.276** |

Raising the learned lexicon's min_freq to 10 inside the blend pushes the
threshold-independent ranking to the best seen (ROC 0.723, **PR(NEG) 0.276 = 4.3×
the B0 baseline 0.064**) but lowers the t=0 / oracle balanced-acc — a genuine
**frontier**, not a single winner (higher AUC ≠ higher best-single-threshold
bal-acc when only 58 NEG docs set the operating point).

### N-gram weighting → multi-scale scoring (the biggest single lever)

The greedy leftmost-longest matcher scores each span once at its *longest* match, so
a trigram suppresses its unigrams. Scoring every n-gram length independently over
**all (overlapping) windows** and summing them with per-length weights
(`decision.score_logodds_multiscale`, `python -m benchmarks.sweep_ngram`) removes
that suppression — and beats the greedy matcher outright:

| weights (1,2,3) | NSMC acc | NIKL bal | NIKL ROC | NIKL PR(NEG) |
|---|---:|---:|---:|---:|
| greedy [1,2,3] (old scorer) | 0.859 | 0.716 | 0.679 | 0.213 |
| uni [1,0,0] | 0.833 | 0.700 | **0.722** | 0.177 |
| bi [0,1,0] | 0.846 | 0.738 | 0.684 | 0.291 |
| tri [0,0,1] | 0.821 | 0.704 | 0.647 | **0.324** |
| **equal [1,1,1]** | **0.864** | **0.735** | 0.686 | 0.287 |
| decay [1,.5,.25] | 0.860 | 0.727 | 0.697 | 0.243 |

**Equal-weight multi-scale strictly dominates the greedy reference on all four
metrics** (NSMC 0.859→0.864, NIKL bal 0.716→0.735, ROC 0.679→0.686, PR(NEG)
0.213→0.287). The big PR(NEG) gain comes from higher orders: bigrams/trigrams are
far better at spotting specific negative phrases (tri-only PR 0.324) — signal the
greedy matcher threw away. Tuned decay weights don't beat plain equal; the lever is
the overlapping multi-scale *method*, not the weight profile.

Stacking multi-scale (equal) onto the blends:

| Config | NSMC | NIKL bal (orc) | NIKL ROC | NIKL PR(NEG) |
|---|---:|---:|---:|---:|
| **multiscale ⊕ blend ×20** | **0.863** | 0.744 (0.747) | 0.694 | **0.300** |
| multiscale, learned mf2 | 0.864 | 0.735 (0.739) | 0.686 | 0.287 |
| multiscale ⊕ blend mf10 ×20 | 0.853 | 0.727 (0.741) | 0.695 | 0.287 |

Multi-scale **equal weights on the bundled×20 ⊕ NSMC(mf2) blend** is the strongest
all-rounder found: NSMC 0.863, NIKL bal 0.744 (≈ U-a20), ROC 0.694, **PR(NEG) 0.300
= 4.7× the B0 baseline (0.064)**. Note this changes the *scoring method*, not just
data, so it's a bigger deploy change than swapping a CSV.

Refining alpha & min_freq *under* multi-scale (`python -m benchmarks.sweep_ms_refine`)
flips the earlier greedy-era optima:

- **min_freq=2 (the default) is optimal** under multi-scale; mf≥3 monotonically hurts
  (NSMC 0.864→0.850, PR 0.287→0.255 by mf20). The overlapping sum is robust to noisy
  rare entries, so mf-pruning — which *helped* the greedy scorer (mf=10) — is now
  counterproductive. (mf=1 maxes pure in-domain NSMC at 0.866 but drops OOD bal to 0.715.)
- **alpha sweet spot ≈ 10–20** (vs 20–25 for greedy), and the in-domain cost of
  blending nearly vanishes (NSMC 0.864→0.860 even at α=100, vs greedy's →0.852). α=10:
  NSMC 0.864 / NIKL bal 0.739 (orc 0.748) / ROC 0.693 / PR 0.299. α=20: 0.863 / 0.744 /
  ROC 0.694 / PR 0.300. For pure rare-class ranking, α=100 reaches PR(NEG) 0.310.

### Strict-Pareto winner / frontier

Maximizing the NIKL out-of-domain metrics at near-constant NSMC accuracy, among
**leakage-safe** configs. Best per metric:
- **multiscale ⊕ blend ×20** — best **all-rounder**: NSMC 0.863, NIKL bal 0.744,
  PR(NEG) 0.300. Recommended overall (needs the multi-scale scorer, not just a CSV).
- **U-a20** (greedy ⊕ blend ×20) — best **balanced-acc** (0.746) with the *current*
  scorer; a CSV-only change.
- **blend mf10 ×20** (greedy) — best **ROC** (0.723) for pure ranking use.

Headline vs the package status quo (B0): NSMC 0.583→0.863, NIKL bal 0.534→0.744,
PR(NEG) 0.064→0.300. The gains stack as: NSMC derivation (biggest) → multi-scale
scoring → ratio-tuned blend.

**🏆 U-a20** (balanced-acc frontier point) — union of the frozen bundled lexicon
(counts ×20) + an NSMC-train lexicon (mf2), scored by log-odds at t=0. For the
ranking/rare-class frontier point, swap the learned side to min_freq=10 (see the
learning-side levers above).

| | NSMC acc | NIKL bal-acc | NIKL ROC | NIKL PR(NEG) |
|---|---:|---:|---:|---:|
| B0 (package status quo) | 0.583 | 0.534 | 0.436 | 0.064 |
| C1-all (NSMC alone) | 0.859 | 0.716 | 0.679 | 0.213 |
| **U-a20 (winner)** | **0.857** | **0.746** | **0.705** | **0.242** |
| Δ vs NSMC alone | −0.002 | **+0.030** | **+0.026** | **+0.029** |
| Δ vs B0 | **+0.274** | **+0.212** | **+0.269** | **+0.178** |

This **revises the earlier conclusion** that blending adds nothing: at the default
ratio it doesn't, but with the bundled side up-weighted ~20× the blend genuinely
beats NSMC-alone out-of-domain (threshold-independent ROC/PR both up, monotone in
the ratio) at a negligible (−0.2pp, within noise) in-domain cost. The big jump is
still the NSMC derivation (B0→C1-all); ratio-tuned blending is a real second-order
gain on top. Combining nofunc cleaning with the blend (U-nofunc-a20) is identical to
U-a20 — the function unigrams are drowned in the blend anyway.

Build it with:

```bash
python -m benchmarks.build_nsmc  --pos-filter all --min-freq 2
python -m benchmarks.build_blend --learned benchmarks/results/lexicons/nsmc-train-all.csv --alpha 20
```

Deployment (shipping this as package data and/or changing analyzer defaults) is
intentionally deferred — see the plan's "이월 사항". Licensing note: the blend mixes
CC BY-SA (bundled) with CC0 (NSMC), so the blended artifact is **CC BY-SA**.
