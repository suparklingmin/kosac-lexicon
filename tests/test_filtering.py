import kosac


def test_set_lexicon_min_freq_shrinks_and_filters():
    lex = kosac.load_lexicon("polarity", ngrams=[1])
    full = lex.get_size()
    lex.set_lexicon(min_freq=5)
    assert lex.get_size() < full
    assert (lex.get_lexicon()["freq"] >= 5).all()


def test_load_forwards_filter_arguments():
    lex = kosac.load_lexicon("polarity", ngrams=[1], min_freq=5, threshold=0.5)
    df = lex.get_lexicon()
    assert (df["freq"] >= 5).all()
    assert (df["max.prop"] > 0.5).all()
