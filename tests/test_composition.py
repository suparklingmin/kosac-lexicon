import pytest

import kosac
from kosac.tokenizers import Tokenizer


def _probs(analyzer, text):
    return analyzer.analyze(text)["features"]["polarity"]["probs"]


def test_negation_flips_polarity_mass():
    # 좋/VA is positive; placing a negation marker next to it should move mass
    # from POS toward NEG. Base tokenizer + pre-tagged text => no Kiwi needed.
    plain = kosac.SentimentAnalyzer("polarity", tokenizer=Tokenizer(), ngrams=[1])
    negating = kosac.SentimentAnalyzer(
        "polarity", tokenizer=Tokenizer(), ngrams=[1], negation=True, window=2)

    text = "안/MAG 좋/VA"
    p_plain, p_neg = _probs(plain, text), _probs(negating, text)
    assert p_neg["NEG"] > p_plain["NEG"]
    assert p_neg["POS"] < p_plain["POS"]

    # the positive word is flagged as negated in the match metadata
    negated = {m["entry"]: m["negated"]
               for m in negating.analyze(text)["features"]["polarity"]["matches"]}
    assert negated["좋/VA"] is True


def test_intensifier_sharpens_dominant_label():
    plain = kosac.SentimentAnalyzer("polarity", tokenizer=Tokenizer(), ngrams=[1])
    boosting = kosac.SentimentAnalyzer(
        "polarity", tokenizer=Tokenizer(), ngrams=[1],
        intensifier=True, intensifier_factor=3.0, window=2)

    text = "정말/MAG 좋/VA"
    assert _probs(boosting, text)["POS"] > _probs(plain, text)["POS"]


def test_composition_off_by_default_matches_plain():
    a = kosac.SentimentAnalyzer("polarity", tokenizer=Tokenizer(), ngrams=[1])
    assert a.negation is False and a.intensifier is False


def test_negation_end_to_end_with_kiwi():
    pytest.importorskip("kiwipiepy")
    plain = kosac.SentimentAnalyzer("polarity")
    negating = kosac.SentimentAnalyzer("polarity", negation=True)
    text = "이 영화는 안 좋다"
    assert _probs(negating, text)["NEG"] > _probs(plain, text)["NEG"]
