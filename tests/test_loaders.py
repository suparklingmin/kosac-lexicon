import pytest

import kosac
from kosac.lexicon import SentimentLexicon

EXPECTED_LABELS = {
    "polarity": ["COMP", "NEG", "NEUT", "None", "POS"],
    "intensity": ["High", "Low", "Medium", "None"],
    "expressive-type": ["dir-action", "dir-explicit", "dir-speech", "indirect", "writing-device"],
    "nested-order": ["0", "1", "2", "3"],
    "subjectivity-polarity": ["COMP", "NEG", "NEUT", "POS"],
    "subjectivity-type": ["Agreement", "Argument", "Emotion", "Intention", "Judgment", "Others", "Speculation"],
}


@pytest.mark.parametrize("feature", kosac.FEATURES)
def test_load_each_bundled_feature(feature):
    lex = kosac.load_lexicon(feature, ngrams=[1])
    assert isinstance(lex, SentimentLexicon)
    assert lex.get_labels() == EXPECTED_LABELS[feature]

    df = lex.get_lexicon()
    assert df.index.name == "entry"
    for column in ("freq", "max.value", "max.prop"):
        assert column in df.columns
    assert lex.get_size() > 0


def test_polarity_unigram_count():
    lex = kosac.load_lexicon("polarity", ngrams=[1])
    assert lex.get_size() == 3476


def test_ngram_selection_is_monotonic():
    sizes = [kosac.load_lexicon("polarity", ngrams=ns).get_size()
             for ns in ([1], [1, 2], [1, 2, 3])]
    assert sizes[0] < sizes[1] < sizes[2]


def test_unknown_feature_raises():
    with pytest.raises(ValueError):
        kosac.load_lexicon("nonsense")
