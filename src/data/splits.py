"""One date-based split used by selection, the index, and training."""
import numpy as np
import pandas as pd
from src.config import DataConfig

def assign_time_split(df, train_end=DataConfig().train_end, val_end=DataConfig().val_end):
    train_end, val_end = pd.Timestamp(train_end), pd.Timestamp(val_end)
    if train_end >= val_end or train_end.weekday() != 6 or val_end.weekday() != 6:
        raise ValueError("Split cutoffs must be increasing Sundays")
    out = df.copy()
    ends = pd.to_datetime(out["year_week"] + "-1", format="%G-W%V-%u") + pd.Timedelta(days=6)
    out["split"] = np.where(ends <= train_end, "train", np.where(ends <= val_end, "val", "test"))
    return out
