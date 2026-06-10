"""Data loaders.

Design intent
-------------
The core library is source-agnostic: loaders return a standard ``pandas.DataFrame``
with a ``DatetimeIndex`` and at least an adjusted close column, regardless of where
the data came from. Raw downloads are cached under ``data/raw/`` and never modified;
cleaned series go to ``data/processed/``.

Open decisions (Phase 1):
- Data source: yfinance (free, convenient, but quality caveats) vs. a curated CSV
  snapshot committed to ``data/raw`` documentation (most reproducible) vs. another API.
- Universe: single index vs. multi-asset.
- Frequency: daily is the recommended starting point.

TODO (Phase 1):
- implement ``load_prices`` with local caching;
- record the exact download date and source in a metadata sidecar for reproducibility.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_prices(
    tickers: list[str],
    start: str | None = None,
    end: str | None = None,
    cache_dir: str | Path = "data/raw",
) -> pd.DataFrame:
    """Load (adjusted) price series for ``tickers`` between ``start`` and ``end``.

    Returns a DataFrame indexed by date with one column per ticker.

    TODO: implement in Phase 1 once the data source is chosen.
    """
    raise NotImplementedError("Data loading is implemented in Phase 1.")
