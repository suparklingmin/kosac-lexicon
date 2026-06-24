import pytest

pytest.importorskip("sklearn")
from kosac.sklearn import KosacVectorizer
from kosac.tokenizers import Tokenizer


def test_vectorizer_shapes_and_feature_names():
    vec = KosacVectorizer("polarity", ngrams=[1], tokenizer=Tokenizer())
    X = vec.fit_transform(["좋/VA", "힘/NNG"])
    names = list(vec.get_feature_names_out())

    assert X.shape == (2, len(names))
    assert names == [f"polarity={label}" for label in ["COMP", "NEG", "NEUT", "None", "POS"]]
    # rows are probability distributions over the feature's labels
    assert all(abs(row.sum() - 1.0) < 1e-9 for row in X)


def test_vectorizer_all_features_width():
    vec = KosacVectorizer("all", ngrams=[1], tokenizer=Tokenizer()).fit(["좋/VA"])
    expected = sum(len(lex.get_labels()) for lex in vec.analyzer_.lexicons.values())
    assert len(vec.get_feature_names_out()) == expected
