from kosac.corpora import Corpus
from kosac.lexicon import GenericLexicon, PolarityLexicon
from kosac.tokenizers import Tokenizer


def test_save_reload_concrete_subclass(tmp_path):
    lex = PolarityLexicon(ngrams=[1])
    lex.update([("좋/VA", "POS"), ("좋/VA", "POS"), ("싫/VA", "NEG")])
    path = tmp_path / "pol.csv"
    lex.save(str(path))

    assert open(path).read().splitlines()[0] == "ngram,COMP,NEG,NEUT,None,POS"
    reloaded = PolarityLexicon(filepath=str(path), ngrams=[1])
    assert reloaded.get_size() == 2
    assert int(reloaded.get_entry("좋/VA")["POS"]) == 2
    assert reloaded.get_entry("싫/VA")["max.value"] == "NEG"


def test_save_reload_generic_infers_labels(tmp_path):
    lex = GenericLexicon(ngrams=[1])
    lex.set_labels(["POS", "NEG"])
    lex.update([("좋/VA", "POS"), ("좋/VA", "POS"), ("싫/VA", "NEG")])
    path = tmp_path / "gen.csv"
    lex.save(str(path))

    reloaded = GenericLexicon(filepath=str(path), ngrams=[1])
    assert reloaded.get_labels() == ["POS", "NEG"]   # inferred from CSV columns
    assert int(reloaded.get_entry("좋/VA")["POS"]) == 2
    assert int(reloaded.get_entry("좋/VA")["freq"]) == 2


def test_build_lexicon_from_empty():
    lex = GenericLexicon(ngrams=[1])
    lex.set_labels(["POS", "NEG"])
    lex.add_token("좋/VA", "POS", verbose=False)
    lex.update([("싫/VA", "NEG"), ("좋/VA", "POS")])

    df = lex.get_lexicon()
    assert set(df.index) == {"좋/VA", "싫/VA"}
    assert int(df.loc["좋/VA", "POS"]) == 2
    assert df.loc["싫/VA", "max.value"] == "NEG"


def _corpus(tmp_path, rows):
    path = tmp_path / "corpus.csv"
    path.write_text("".join(f"{t},{l}\n" for t, l in rows), encoding="utf-8")
    return Corpus(str(path))


def test_update_from_corpus_pos_tag_drops_function_words(tmp_path):
    # Pre-tagged text + the whitespace Tokenizer keeps this Kiwi-free.
    corpus = _corpus(tmp_path, [("이/MM 제품/NNG 좋/VA", "POS"),
                                ("서비스/NNG 나쁘/VA 다/EF", "NEG")])
    lex = GenericLexicon(ngrams=[1])
    lex.set_labels(["POS", "NEG"])
    lex.update_from_corpus(corpus, Tokenizer(), pos_tag={"NNG", "VV", "VA"})

    entries = set(lex.get_lexicon().index)
    assert "이/MM" not in entries and "다/EF" not in entries
    assert {"제품/NNG", "좋/VA", "서비스/NNG", "나쁘/VA"} <= entries


def test_update_from_corpus_min_freq(tmp_path):
    corpus = _corpus(tmp_path, [("좋/VA", "POS"), ("좋/VA", "POS"), ("싫/VA", "NEG")])
    lex = GenericLexicon(ngrams=[1])
    lex.set_labels(["POS", "NEG"])
    lex.update_from_corpus(corpus, Tokenizer(), min_freq=2)
    assert set(lex.get_lexicon().index) == {"좋/VA"}


def test_update_from_corpus_max_value_threshold(tmp_path):
    # 좋/VA: POS, POS, NEG -> max.prop = 2/3 ≈ 0.67, below 0.7 -> dropped.
    corpus = _corpus(tmp_path, [("좋/VA", "POS"), ("좋/VA", "POS"), ("좋/VA", "NEG")])
    lex = GenericLexicon(ngrams=[1])
    lex.set_labels(["POS", "NEG"])
    lex.update_from_corpus(corpus, Tokenizer(), max_value_threshold=0.7)
    assert lex.get_size() == 0


def test_update_from_corpus_no_filters_is_unchanged(tmp_path):
    corpus = _corpus(tmp_path, [("이/MM 좋/VA", "POS")])
    lex = GenericLexicon(ngrams=[1])
    lex.set_labels(["POS", "NEG"])
    lex.update_from_corpus(corpus, Tokenizer())
    assert set(lex.get_lexicon().index) == {"이/MM", "좋/VA"}


def test_update_from_corpus_pos_tag_keeps_mixed_ngrams(tmp_path):
    # An N-gram with a content token survives even if other tokens are function
    # morphemes — e.g. ㄹ/ETM 수/NNB 있/VV ("can ...").
    corpus = _corpus(tmp_path, [("ㄹ/ETM 수/NNB 있/VV", "POS")])
    lex = GenericLexicon(ngrams=[1, 2, 3])
    lex.set_labels(["POS", "NEG"])
    lex.update_from_corpus(corpus, Tokenizer(), pos_tag={"NNG", "VV", "VA"})

    entries = set(lex.get_lexicon().index)
    assert "ㄹ/ETM 수/NNB 있/VV" in entries   # kept (contains 있/VV)
    assert "있/VV" in entries
    assert "ㄹ/ETM" not in entries            # pure function unigram dropped
    assert "ㄹ/ETM 수/NNB" not in entries     # no content token -> dropped
