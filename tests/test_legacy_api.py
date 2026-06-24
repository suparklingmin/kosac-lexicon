import kosac
from kosac.tokenizers import Tokenizer


def test_set_lexicon_can_be_loosened_again():
    # Re-applying set_lexicon filters the original lexicon, so the threshold can
    # be loosened back to the full set (as in the 2024-01-18 legacy notebook).
    lex = kosac.load_lexicon("polarity", ngrams=[1])
    full = lex.get_size()
    lex.set_lexicon(min_freq=5)
    assert lex.get_size() < full
    lex.set_lexicon(min_freq=1)
    assert lex.get_size() == full


def test_legacy_aliases_get_and_match():
    lex = kosac.load_lexicon("polarity", ngrams=[1])
    # `get` is an alias of `get_entry`
    assert lex.get("힘/NNG").equals(lex.get_entry("힘/NNG"))
    # `match` is an alias of `match_patterns`
    matched = lex.match("힘/NNG 좋/VA", Tokenizer())
    assert set(matched) == set(lex.match_patterns("힘/NNG 좋/VA", Tokenizer()))
