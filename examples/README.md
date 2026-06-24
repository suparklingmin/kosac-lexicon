# Examples

- [`quickstart.py`](quickstart.py) — runnable end-to-end demo using the current
  `kosac` package API (`kosac.load_lexicon(...)`). Start here.
- `demo-2024*.ipynb` — notebooks from development using the `kosac` package.
  They were written against the in-repo `./lexicon/` and `./data/` paths; load
  the bundled data with `kosac.load_lexicon(...)` instead now that those files
  live inside the package.
- [`legacy/`](legacy/) — the original 2024 monolithic prototype (`draft.py`) and
  the notebooks that import it via `from utils import *`. Kept for provenance;
  they predate the `kosac` package and do not reflect the current API.
