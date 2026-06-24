from kosac.lexicon import GenericLexicon


def test_build_lexicon_from_empty():
    lex = GenericLexicon(ngrams=[1])
    lex.set_labels(["POS", "NEG"])
    lex.add_token("좋/VA", "POS", verbose=False)
    lex.update([("싫/VA", "NEG"), ("좋/VA", "POS")])

    df = lex.get_lexicon()
    assert set(df.index) == {"좋/VA", "싫/VA"}
    assert int(df.loc["좋/VA", "POS"]) == 2
    assert df.loc["싫/VA", "max.value"] == "NEG"
