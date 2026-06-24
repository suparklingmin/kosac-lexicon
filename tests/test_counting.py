import kosac
from kosac.tokenizers import Tokenizer


def test_count_tallies_matches_by_max_value():
    # Base tokenizer + pre-tagged text => deterministic, no Kiwi needed.
    analyzer = kosac.SentimentAnalyzer("polarity", tokenizer=Tokenizer(), ngrams=[1])
    counted = analyzer.count("힘/NNG 좋/VA")["features"]["polarity"]

    assert counted["total"] == 2
    assert sum(counted["counts"].values()) == 2
    # both 힘/NNG and 좋/VA are POS-dominant in the bundled lexicon
    assert counted["counts"]["POS"] == 2
    assert counted["label"] == "POS"
    assert abs(sum(counted["proportions"].values()) - 1.0) < 1e-9


def test_count_frame_has_per_label_columns():
    analyzer = kosac.SentimentAnalyzer("polarity", tokenizer=Tokenizer(), ngrams=[1])
    frame = analyzer.count_frame(["좋/VA", "힘/NNG"])
    assert {"polarity.label", "polarity.total", "polarity.POS", "polarity.NEG"} <= set(frame.columns)


def test_count_empty_text_has_no_matches():
    analyzer = kosac.SentimentAnalyzer("polarity", tokenizer=Tokenizer(), ngrams=[1])
    counted = analyzer.count("없는단어/ZZ")["features"]["polarity"]
    assert counted["total"] == 0 and counted["label"] is None
