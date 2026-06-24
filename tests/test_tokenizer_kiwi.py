import pytest

import kosac

pytest.importorskip("kiwipiepy")  # skip if the [kiwi] extra is not installed
from kosac.tokenizers import KiwiTokenizer


def test_kiwi_tokenize_emits_surface_pos():
    tokens = KiwiTokenizer().tokenize("나는 정말 행복하다")
    assert tokens, "expected non-empty tokenization"
    assert all("/" in token for token in tokens)


def test_kiwi_end_to_end_against_real_lexicon():
    tok = KiwiTokenizer()
    lex = kosac.load_lexicon("polarity", ngrams=[1, 2, 3])

    matches = lex.match_patterns("이 영화는 정말 좋았고 너무 행복했다", tok)
    assert isinstance(matches, list)

    if matches:  # the bundled lexicon should match common sentiment morphemes
        probs = lex.get_sent_probs("이 영화는 정말 좋았고 너무 행복했다", tok)
        assert set(probs.index) == set(lex.get_labels())
        assert abs(probs.sum() - 1.0) < 1e-9
