import pytest

from kosac.lexicon import PolarityLexicon

# A tiny inline polarity lexicon used by the matching/inference tests.
# Columns mirror the real CSVs: ngram + absolute counts only; freq, max.value,
# and max.prop are derived by the loader. The '가*/JKS' row exercises regex chars.
MINI_CSV = """ngram,COMP,NEG,NEUT,None,POS
힘/NNG,0,7,1,0,2
좋/VA,0,0,0,0,5
가*/JKS,0,0,0,0,3
"""


@pytest.fixture
def mini_csv(tmp_path):
    path = tmp_path / "mini-polarity.csv"
    path.write_text(MINI_CSV, encoding="utf-8")
    return str(path)


@pytest.fixture
def mini_lexicon(mini_csv):
    return PolarityLexicon(filepath=mini_csv, ngrams=[1])
