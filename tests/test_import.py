import kosac


def test_import_exposes_api():
    assert len(kosac.FEATURES) == 6
    assert "polarity" in kosac.FEATURES
    assert callable(kosac.load_lexicon)
    assert kosac.__version__
    assert kosac.__data_version__ == "2016"


def test_importing_tokenizers_needs_no_extras():
    # The heavy kiwi/transformers imports must be lazy, so importing the
    # module (and referencing the classes) works without those extras.
    import kosac.tokenizers as tk

    assert hasattr(tk, "KiwiTokenizer")
    assert hasattr(tk, "HuggingFaceTokenizer")
