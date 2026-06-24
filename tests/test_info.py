import kosac


def test_describe_feature_values_match_lexicon_labels():
    for feature in kosac.FEATURES:
        described = set(kosac.describe_feature(feature)["values"])
        labels = set(kosac.load_lexicon(feature, ngrams=[1]).get_labels())
        assert described == labels


def test_describe_unknown_feature_raises():
    import pytest
    with pytest.raises(ValueError):
        kosac.describe_feature("nope")


def test_citation_is_bibtex():
    bib = kosac.citation()
    assert bib.startswith("@misc{")
    assert kosac.__version__ in bib
