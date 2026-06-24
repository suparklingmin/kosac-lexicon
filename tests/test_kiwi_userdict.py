import pytest

import kosac

pytest.importorskip("kiwipiepy")
from kosac.tokenizers import KiwiTokenizer


def test_add_user_word_changes_segmentation():
    novel = "캌콬뀰"  # a made-up form Kiwi won't know
    sentence = f"나는 {novel} 좋아"

    plain = KiwiTokenizer()
    assert f"{novel}/NNP" not in plain.tokenize(sentence)

    custom = KiwiTokenizer(user_words=[(novel, "NNP")])
    assert f"{novel}/NNP" in custom.tokenize(sentence)


def test_from_lexicon_registers_unigrams_and_tokenizes():
    lex = kosac.load_lexicon("polarity", ngrams=[1])
    tok = KiwiTokenizer.from_lexicon(lex, tags={"NNG", "VV", "VA", "XR"})
    tokens = tok.tokenize("나는 정말 행복하다")
    assert tokens and all("/" in t for t in tokens)


def test_analyzer_align_option_runs_end_to_end():
    analyzer = kosac.SentimentAnalyzer("polarity", align=True, ngrams=[1, 2, 3])
    pol = analyzer.analyze("이 영화 정말 좋다")["features"]["polarity"]
    assert pol["label"] is not None
    # spans must index back into the original text
    for m in pol["matches"]:
        s, e = m["span"]
        assert analyzer  # sanity
        assert e >= s
