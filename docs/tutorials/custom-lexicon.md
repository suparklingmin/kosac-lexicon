# Building a custom lexicon

The bundled lexicons are fixed (2016 data), but you can **extend** them with new
words or **build your own** from labeled examples. This uses the lower-level
lexicon API (the `SentimentAnalyzer` loads bundled features by name).

## Extending an existing lexicon

Load a lexicon and inspect an entry. In KOSAC, `귀엽/VA` ("cute") happens to
appear once, negatively:

```python
import kosac

lex = kosac.load_lexicon("polarity", ngrams=[1])
lex.get_entry("귀엽/VA").to_dict()
# {'ngram': 1, 'freq': 1, 'COMP': 0, 'NEG': 1, 'NEUT': 0, 'None': 0, 'POS': 0,
#  'max.value': 'NEG', 'max.prop': 1.0}
```

`add_token(morph, label)` records one new observation and re-derives `max.value`
/ `max.prop` automatically:

```python
lex.add_token("귀엽/VA", "POS", verbose=False)
lex.get_entry("귀엽/VA").to_dict()
# {... 'freq': 2, 'NEG': 1, 'POS': 1, 'max.value': 'NEG', 'max.prop': 0.5}
```

(With a tie, `max.value` keeps the first label in column order.)

Adding a brand-new entry works the same way — here a neologism that isn't in
KOSAC:

```python
"커엽/VA" in lex.get_lexicon().index        # False
lex.add_token("커엽/VA", "POS", verbose=False)
lex.get_entry("커엽/VA").to_dict()
# {'ngram': 1, 'freq': 1, 'POS': 1, 'max.value': 'POS', 'max.prop': 1.0}
```

Add several observations at once with `update()`:

```python
lex.update([("귀엽/VA", "POS"), ("귀엽/VA", "NEG"), ("커엽/VA", "POS")])
lex.get_entry("귀엽/VA").to_dict()
# {... 'freq': 4, 'NEG': 2, 'POS': 2, 'max.value': 'NEG', 'max.prop': 0.5}
```

## Building from scratch

Start from an empty `GenericLexicon` and declare your own labels:

```python
from kosac.lexicon import GenericLexicon

lex = GenericLexicon(ngrams=[1])
lex.set_labels(["POS", "NEG"])
lex.update([("좋/VA", "POS"), ("좋/VA", "POS"), ("싫/VA", "NEG")])

lex.get_lexicon()[["freq", "POS", "NEG", "max.value", "max.prop"]]
```

```text
      freq POS NEG max.value max.prop
entry
좋/VA      2   2   0       POS      1.0
싫/VA      1   0   1       NEG      1.0
```

## Building from a labeled corpus

`Corpus` reads a headerless `text,label` CSV, and `update_from_corpus` tokenizes
each text and assigns its label to every morpheme N-gram:

```python
from kosac.corpora import Corpus
from kosac.tokenizers import KiwiTokenizer

# mini.csv:
#   이 제품 정말 좋다,POS
#   서비스가 너무 별로다,NEG
corpus = Corpus("mini.csv")
corpus.get_labels()          # ['POS', 'NEG']

lex = GenericLexicon(ngrams=[1])
lex.set_labels(["POS", "NEG"])
lex.update_from_corpus(corpus, KiwiTokenizer())

lex.get_size()               # 10
lex.get_lexicon()[["freq", "POS", "NEG", "max.value"]].head()
```

```text
        freq POS NEG max.value
entry
이/MM       1   1   0       POS
제품/NNG     1   1   0       POS
정말/MAG     1   1   0       POS
좋/VA       1   1   0       POS
다/EF       2   1   1       POS
```

Every morpheme of a `POS` sentence is counted as POS, so function morphemes
(`이/MM`, `다/EF`, `가/JKS`, …) are aggregated too — the same noise discussed in
the [counting tutorial](counting.md). Restrict the build to **content words**
with `pos_tag`:

```python
lex.update_from_corpus(
    corpus, KiwiTokenizer(),
    pos_tag={"NNG", "NNP", "VV", "VA", "XR", "MAG"},   # content words only
)
list(lex.get_lexicon().index)
# ['제품/NNG', '정말/MAG', '좋/VA', '서비스/NNG', '너무/MAG', '별로/MAG']
```

`min_freq` and `max_value_threshold` drop rare or low-confidence entries as well,
e.g. `update_from_corpus(corpus, tok, pos_tag=..., min_freq=5,
max_value_threshold=0.6)`.

## Using your custom lexicon

A custom lexicon supports the same matching and scoring as the bundled ones —
pass it a tokenizer:

```python
tok = KiwiTokenizer()
lex.match_patterns("이 제품은 좋다", tok)
# ['이/MM', '제품/NNG', '좋/VA', '다/EF']

lex.get_sent_probs("이 제품은 좋다", tok).round(3).to_dict()
# {'POS': 0.889, 'NEG': 0.111}
```

You can also export the unigrams as a Kiwi/Komoran user dictionary:

```python
lex.export_user_dict("user_dictionary.txt")   # one 'surface\tPOS' line per unigram
```

## Saving and reloading

The constructor's CSV loader expects *relative* frequencies (it multiplies them
by `freq` on load), so the simplest lossless way to persist a built lexicon is to
pickle the object:

```python
import pickle

with open("lex.pkl", "wb") as f:
    pickle.dump(lex, f)

with open("lex.pkl", "rb") as f:
    reloaded = pickle.load(f)

reloaded.get_size()                  # 2
reloaded.get_entry("좋/VA")[["freq", "POS", "NEG", "max.value"]].to_dict()
# {'freq': 2, 'POS': 2, 'NEG': 0, 'max.value': 'POS'}
```

