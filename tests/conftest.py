import pytest

from kosac.lexicon import PolarityLexicon

# A tiny inline polarity lexicon used by the matching/inference tests.
# Columns mirror the real CSVs; the '가*/JKS' row exercises regex-special chars.
MINI_CSV = """ngram,freq,COMP,NEG,NEUT,None,POS,max.value,max.prop
힘/NNG,10,0.0,0.7,0.1,0.0,0.2,NEG,0.7
좋/VA,5,0.0,0.0,0.0,0.0,1.0,POS,1.0
가*/JKS,3,0.0,0.0,0.0,0.0,1.0,POS,1.0
"""


@pytest.fixture
def mini_csv(tmp_path):
    path = tmp_path / "mini-polarity.csv"
    path.write_text(MINI_CSV, encoding="utf-8")
    return str(path)


@pytest.fixture
def mini_lexicon(mini_csv):
    return PolarityLexicon(filepath=mini_csv, ngrams=[1])
