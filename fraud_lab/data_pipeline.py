from __future__ import annotations

import pandas as pd

from backend.ml.engine import load_or_create_dataset


def load_main_dataset(force_reload: bool = False) -> pd.DataFrame:
    """
    Load the primary fraud dataset used across Streamlit pages.
    Falls back to the engine's synthetic dataset if none is present.
    """
    return load_or_create_dataset(force_reload=force_reload).copy()


# Backward-compatible alias if needed elsewhere
def load_cached_dataset() -> pd.DataFrame:
    return load_main_dataset(force_reload=False)
