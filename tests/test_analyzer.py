import pytest

import kosac
from kosac.analyzer import select_matches, select_matches_multiscale
from kosac.tokenizers import Tokenizer


def test_select_matches_is_leftmost_longest_and_nonoverlapping():
    # tokens: (str, start, end); prefer the bigram over the two unigrams.
    tokens = [("a", 0, 1), ("b", 2, 3), ("c", 4, 5)]
    out = select_matches(tokens, {"a", "b", "a b", "c"}, [1, 2])
    entries = [m[0] for m in out]
    assert entries == ["a b", "c"]
    # spans come from the first/last token of each match
    assert out[0][3] == 0 and out[0][4] == 3


def test_select_matches_multiscale_keeps_overlaps():
    # multiscale yields every matching window; greedy drops the overlapped unigrams.
    tokens = [("a", 0, 1), ("b", 2, 3), ("c", 4, 5)]
    entry_set = {"a", "b", "a b", "b c"}
    ms = {m[0] for m in select_matches_multiscale(tokens, entry_set, [1, 2])}
    assert ms == {"a", "b", "a b", "b c"}
    greedy = {m[0] for m in select_matches(tokens, entry_set, [1, 2])}
    assert greedy == {"a b"}  # leftmost-longest consumes a,b -> only the bigram


def test_scoring_default_is_multiscale_with_greedy_option():
    a = kosac.SentimentAnalyzer("polarity", tokenizer=Tokenizer(), ngrams=[1])
    assert a.scoring == "multiscale"
    g = kosac.SentimentAnalyzer("polarity", tokenizer=Tokenizer(), scoring="greedy")
    assert g.scoring == "greedy"
    with pytest.raises(ValueError):
        kosac.SentimentAnalyzer("polarity", tokenizer=Tokenizer(), scoring="bogus")


def test_polarity_blend_loads_but_is_not_a_canonical_feature():
    lex = kosac.load_lexicon("polarity-blend", ngrams=[1])
    assert lex.get_labels() == ["NEG", "POS"]
    assert lex.get_size() > 1000
    # derived perf lexicon: loadable by name, but not in FEATURES / analyzer('all')
    assert "polarity-blend" not in kosac.FEATURES


def test_polarity_api_shape():
    a = kosac.SentimentAnalyzer("polarity", tokenizer=Tokenizer(), ngrams=[1])
    assert a.predict_polarity("좋/VA") in {"POS", "NEG"}
    assert -1.0 <= a.polarity_score("좋/VA") <= 1.0
    assert a.predict_polarity_batch(["좋/VA", "힘/NNG"]) == \
        [a.predict_polarity("좋/VA"), a.predict_polarity("힘/NNG")]


def test_polarity_blend_predicts_sentiment_with_kiwi():
    pytest.importorskip("kiwipiepy")
    a = kosac.SentimentAnalyzer("polarity-blend")  # multiscale default
    assert a.predict_polarity("이 영화 정말 재미있고 최고였다") == "POS"
    assert a.predict_polarity("시간 낭비 최악의 영화 너무 지루하다") == "NEG"


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
