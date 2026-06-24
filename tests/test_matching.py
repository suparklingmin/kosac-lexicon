from kosac.tokenizers import Tokenizer


def test_match_patterns_finds_entries(mini_lexicon):
    matches = mini_lexicon.match_patterns("힘/NNG 좋/VA", Tokenizer())
    assert set(matches) == {"힘/NNG", "좋/VA"}


def test_wildcard_entry_is_escaped(mini_lexicon):
    # '가*/JKS' has a regex-special '*'; get_pattern must escape it so the
    # match is literal and re.compile does not raise (regression test).
    matches = mini_lexicon.match_patterns("가*/JKS", Tokenizer())
    assert matches == ["가*/JKS"]


def test_get_match_info_returns_max_values(mini_lexicon):
    # get_match_info used to call a non-existent self.match (regression test).
    info = {m: v for (m, v, _prop) in mini_lexicon.get_match_info("힘/NNG 좋/VA", Tokenizer())}
    assert info["힘/NNG"] == "NEG"
    assert info["좋/VA"] == "POS"
