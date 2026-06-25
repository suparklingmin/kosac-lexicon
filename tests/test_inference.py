from kosac.tokenizers import Tokenizer


def test_sent_probs_form_a_distribution(mini_lexicon):
    # get_sent_probs used to call a non-existent self.match (regression test).
    probs = mini_lexicon.get_sent_probs("힘/NNG 좋/VA", Tokenizer())
    assert set(probs.index) == set(mini_lexicon.get_labels())
    assert abs(probs.sum() - 1.0) < 1e-9


def test_sent_probs_pick_dominant_label(mini_lexicon):
    probs = mini_lexicon.get_sent_probs("좋/VA", Tokenizer())
    assert probs.idxmax() == "POS"


def test_sent_probs_no_match_is_uniform(mini_lexicon):
    # A sentence with no lexicon match has no evidence -> uniform, not a crash.
    probs = mini_lexicon.get_sent_probs("없는단어/NNG", Tokenizer())
    assert set(probs.index) == set(mini_lexicon.get_labels())
    assert abs(probs.sum() - 1.0) < 1e-9
    assert probs.nunique() == 1
