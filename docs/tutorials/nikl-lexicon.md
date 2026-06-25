# Case study: a 3-class lexicon from 모두의 말뭉치 (NIKL) expressions

The [NSMC case study](nsmc-case-study.md) builds a **binary** lexicon from
documents labeled positive / negative. Many sentiment corpora instead label
short **expressions** (phrases) with a numeric *polarity score*. A common shape —
e.g. a 모두의 말뭉치 (NIKL) sentiment-analysis export such as
`EXSA2002108040.csv` — is a data frame with an `expression_form` column (the
phrase) and an integer `expression_score` in −2..2:

| expression_form | expression_score |
|---|---|
| 정말 마음에 쏙 들어요 | 2 |
| 생각보다 괜찮네요 | 1 |
| 그냥 보통이에요 | 0 |
| 조금 아쉬워요 | -1 |
| 완전 실망했어요 | -2 |

```{note}
The rows above are **invented** for illustration. The expression text in a real
corpus may be licensed or private — keep it on your machine; nothing below needs
to publish it, only the derived morpheme lexicon.
```

The same [custom-lexicon](custom-lexicon.md) machinery turns this into a morpheme
polarity lexicon — the only new step is collapsing the **−2..2 score into three
classes**, giving a `POS` / `NEUT` / `NEG` lexicon (KOSAC's own polarity labels
minus `COMP`/`None`).
[`examples/nikl_lexicon.py`](https://github.com/suparklingmin/kosac-lexicon/blob/main/examples/nikl_lexicon.py)
does all of this as a runnable script.

## Mapping score → label

`Corpus` reads a headerless `text,label` CSV, so map the score to a class and
write the two columns. Positive scores are `POS`, negative `NEG`, zero `NEUT`:

```python
import pandas as pd

def score_to_label(score):
    if score > 0:
        return 'POS'
    if score < 0:
        return 'NEG'
    return 'NEUT'

df = pd.read_csv('EXSA2002108040.csv')        # your 모두의 말뭉치 export
df['expression_form'] = df['expression_form'].astype(str).str.strip()
df = df[df['expression_form'] != '']
df['label'] = df['expression_score'].astype(int).map(score_to_label)
df[['expression_form', 'label']].to_csv('expr_corpus.csv', index=False, header=False)
```

```{note}
Real expression corpora are usually **class-imbalanced** — review data skews
positive (here ≈ 16k POS / 2.7k NEG / 0.8k NEUT across product, movie, and travel
reviews). That imbalance carries into the lexicon and biases sentence scores
toward the majority class; see the caveat at the end.
```

## Building

`update_from_corpus` tokenizes each expression into morpheme N-grams and counts
its label for every N-gram. Declare the **three** labels, restrict to content
words with `pos_tag`, and drop rare entries with `min_freq`:

```python
from kosac.corpora import Corpus
from kosac.lexicon import GenericLexicon
from kosac.tokenizers import KiwiTokenizer

corpus = Corpus('expr_corpus.csv')
corpus.get_labels()        # the three classes (order follows first appearance)

lex = GenericLexicon(ngrams=[1, 2, 3])
lex.set_labels(['POS', 'NEUT', 'NEG'])
lex.update_from_corpus(
    corpus, KiwiTokenizer(),
    pos_tag={'NNG', 'NNP', 'VV', 'VA', 'XR', 'MAG'},   # content words
    min_freq=5,
)
lex.get_size()             # entries seen >= 5 times (uni + bi + tri-grams)
```

## Inspecting

Sort the content unigrams by dominant-label proportion (`max.prop`) for a quick
sanity check — the strongly-polar morphemes should land in the expected column:

```python
df = lex.get_lexicon()
uni = df[(df['ngram'] == 1) & (df['freq'] >= 10)]
uni[uni['max.value'] == 'POS'].sort_values('max.prop', ascending=False).head(10)
```

The columns separate by kind of word. The entries below are the actual unigrams
from this build — a **mixed-domain** review corpus (product, movie, and travel
reviews), so the strongest-polar words carry a domain flavour: skincare terms
from the product reviews (촉촉, 흡수, 제형), film terms from the movie reviews
(유치, 뻔하):

| POS | NEUT | NEG |
|-----|------|-----|
| 촉촉 (moist), 편하 (comfortable), 굿 (good) | 호불호 (hit-or-miss), 궁금 (curious) | 씁쓸 (bitter), 어설프 (clumsy), 싫 (dislike) |
| 안심 (reassuring), 진정 (soothing), 흡수 (absorbs well) | 묽 (watery), 비슷 (similar), 제형 (texture) | 아쉬움 (let-down), 유치 (cheesy), 뻔하 (predictable) |

`NEUT` is the smallest and noisiest class — score-0 expressions are sparse, so its
top entries are mixed or factual words (호불호 "love-it-or-hate-it", 묽 "watery",
제형 "texture") rather than strong sentiment; relax the `freq` filter to surface
them. Bigrams add light constructions the unigrams miss (`도/JX 좋/VA`
"… is also good").

## Scoring

A custom lexicon supports the same matching and scoring as the bundled ones —
`get_sent_probs` is a softmax over the matched morphemes' add-one-smoothed
log-probabilities:

```python
tok = KiwiTokenizer()
lex.get_sent_probs('이 제품 정말 좋고 만족스러워요', tok).round(3).to_dict()
# {'POS': 0.997, 'NEUT': 0.002, 'NEG': 0.002}
lex.get_sent_probs('완전 실망했고 최악이에요', tok).round(3).to_dict()
# {'NEG': 0.557, 'POS': 0.437, 'NEUT': 0.006}
```

A sentence with **no** matching entry has no evidence, so the distribution is
uniform rather than an error:

```python
lex.get_sent_probs('xyz', tok).to_dict()
# {'POS': 0.333, 'NEUT': 0.333, 'NEG': 0.333}
```

```{warning}
Notice the second example only just lands on `NEG` (0.557). With a POS-skewed
corpus, weakly-negative sentences can come out POS: every smoothed entry leaks a
little probability to the majority class, and enough neutral-ish matches outvote
one negative word. If balanced predictions matter more than reproducing the
corpus distribution, tighten the match set (`set_lexicon(min_freq=…,
threshold=…)` to keep only high-`max.prop` entries), down-sample the majority
class before building, or compare summed log-probs only between `POS`/`NEG`.
```

## Saving

`save()` writes the package's absolute-count CSV (`ngram` + one column per
label); the constructor reads it back and a `GenericLexicon` **infers** the
labels from the columns:

```python
lex.save('nikl_lexicon.csv')

reloaded = GenericLexicon(filepath='nikl_lexicon.csv', ngrams=[1, 2, 3])
reloaded.get_labels()        # ['POS', 'NEUT', 'NEG']  (inferred from the columns)
```

The saved CSV holds only the derived morpheme counts, not the source expressions
— so it is safe to share even when the original corpus text is not.

```{note}
This lexicon isn't bundled with the package — you build it from your own
expression-scored data with the snippets above (or
`examples/nikl_lexicon.py`, which falls back to a tiny built-in sample of
invented phrases so it runs with no corpus:
`python examples/nikl_lexicon.py --data EXSA2002108040.csv`).
The package ships only the frozen KOSAC lexicons.
```

## References

> National Institute of Korean Language (2021). NIKL Sentiment Analysis Corpus
> (v.1.0). URL: <https://kli.korean.go.kr/corpus>
