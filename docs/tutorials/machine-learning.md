# Sentiment features for machine learning

`KosacVectorizer` turns Korean text into KOSAC label-probability features, so the
lexicon can feed a scikit-learn pipeline.

## Install

```bash
pip install "kosac-lexicon[kiwi,sklearn]"
```

(`[kiwi]` for the tokenizer, `[sklearn]` for the estimator base classes.)

## A classification pipeline

```python
from kosac.sklearn import KosacVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

train_texts = [
    "정말 훌륭하고 만족스러운 제품이다",
    "최고의 경험이었고 매우 행복했다",
    "친절한 서비스에 기분이 좋았다",
    "형편없고 실망스러운 품질이다",
    "최악이었고 다시는 안 쓴다",
    "불친절해서 너무 화가 났다",
]
labels = ["pos", "pos", "pos", "neg", "neg", "neg"]

clf = make_pipeline(KosacVectorizer("all"), LogisticRegression(max_iter=1000))
clf.fit(train_texts, labels)

clf.predict(["기대 이상으로 만족스럽다", "두 번 다시 가고 싶지 않다"])
# array(['pos', 'neg'], dtype='<U3')
```

`KosacVectorizer("all")` emits one column per `<feature>=<label>` probability —
29 columns across the six features.

## Inspecting the features

```python
vec = KosacVectorizer("polarity").fit(train_texts)
vec.get_feature_names_out()
# array(['polarity=COMP', 'polarity=NEG', 'polarity=NEUT',
#        'polarity=None', 'polarity=POS'], dtype=object)

vec.transform(train_texts).shape   # (6, 5)
```

Each row is the per-label probability distribution from `analyze()` (the
probability method, which handles function-morpheme noise — see the
[social-science tutorial](social-science.md)).

## Tips for real data

- The constructor mirrors `SentimentAnalyzer`: `KosacVectorizer("all",
  negation=True, ngrams=[1, 2, 3], min_freq=5)`.
- Evaluate with cross-validation, not a tiny toy set:

  ```python
  from sklearn.model_selection import cross_val_score
  cross_val_score(clf, texts, labels, cv=5)
  ```

- Combine lexicon features with bag-of-words for a stronger baseline:

  ```python
  from sklearn.feature_extraction.text import TfidfVectorizer
  from sklearn.pipeline import FeatureUnion

  features = FeatureUnion([
      ("kosac", KosacVectorizer("all")),
      ("tfidf", TfidfVectorizer()),
  ])
  ```
