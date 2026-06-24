import pytest

import kosac
from kosac.analyzer import select_matches
from kosac.tokenizers import Tokenizer


def test_select_matches_is_leftmost_longest_and_nonoverlapping():
    # tokens: (str, start, end); prefer the bigram over the two unigrams.
    tokens = [("a", 0, 1), ("b", 2, 3), ("c", 4, 5)]
    out = select_matches(tokens, {"a", "b", "a b", "c"}, [1, 2])
    entries = [m[0] for m in out]
    assert entries == ["a b", "c"]
    # spans come from the first/last token of each match
    assert out[0][3] == 0 and out[0][4] == 3


def test_analyze_polarity_with_pretagged_text():
    # Base tokenizer + pre-tagged input keeps this test Java/Kiwi-free.
    analyzer = kosac.SentimentAnalyzer("polarity", tokenizer=Tokenizer(), ngrams=[1])
    result = analyzer.analyze("힘/NNG 좋/VA")

    pol = result["features"]["polarity"]
    assert pol["label"] in analyzer.lexicons["polarity"].get_labels()
    assert abs(sum(pol["probs"].values()) - 1.0) < 1e-9
    entries = {m["entry"] for m in pol["matches"]}
    assert {"힘/NNG", "좋/VA"} <= entries
    # every match carries a character span into the original text
    for m in pol["matches"]:
        assert result["text"][m["span"][0]:m["span"][1]] == m["text"]


def test_analyze_all_six_features_at_once():
    analyzer = kosac.SentimentAnalyzer("all", tokenizer=Tokenizer(), ngrams=[1])
    result = analyzer.analyze("좋/VA")
    assert set(result["features"]) == set(kosac.FEATURES)


def test_analyze_frame_shape():
    analyzer = kosac.SentimentAnalyzer("polarity", tokenizer=Tokenizer(), ngrams=[1])
    frame = analyzer.analyze_frame(["좋/VA", "힘/NNG"])
    assert list(frame["text"]) == ["좋/VA", "힘/NNG"]
    assert "polarity.label" in frame.columns and "polarity.prob" in frame.columns


def test_analyzer_end_to_end_with_kiwi():
    pytest.importorskip("kiwipiepy")
    analyzer = kosac.SentimentAnalyzer("polarity")  # default Kiwi tokenizer
    pol = analyzer.analyze("이 영화는 정말 좋았고 너무 행복했다")["features"]["polarity"]
    assert pol["label"] is not None
    assert abs(sum(pol["probs"].values()) - 1.0) < 1e-9
