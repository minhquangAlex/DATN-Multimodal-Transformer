"""Aggregate transactions in chunks, preserving the unfiltered weekly table."""
from pathlib import Path
import numpy as np
import pandas as pd
from src.config import write_json


def normalize_year_week(dates):
    series = pd.to_datetime(dates if isinstance(dates, pd.Series) else pd.Series(dates))
    iso = series.dt.isocalendar()
    result = iso.year.astype(str) + "-W" + iso.week.astype(str).str.zfill(2)
    return result if isinstance(dates, pd.Series) else result.tolist()


def week_calendar(first_date, last_date):
    first, last = pd.Timestamp(first_date), pd.Timestamp(last_date)
    monday = first.normalize() - pd.Timedelta(days=first.weekday())
    starts = pd.date_range(monday, last, freq="W-MON")
    frame = pd.DataFrame({"week_start": starts})
    frame["week_end"] = frame.week_start + pd.Timedelta(days=6)
    frame["year_week"] = normalize_year_week(frame.week_start)
    frame["is_complete_week"] = (frame.week_start >= first) & (frame.week_end <= last)
    iso_week = frame.week_start.dt.isocalendar().week.astype(int)
    frame["week_of_year"] = iso_week
    frame["month"] = frame.week_start.dt.month
    frame["week_sin"] = np.sin(2 * np.pi * (iso_week - 1) / 52.1775)
    frame["week_cos"] = np.cos(2 * np.pi * (iso_week - 1) / 52.1775)
    return frame


def build_weekly_sales(raw_data_dir="raw_data", processed_data_dir="processed_data/v2", chunksize=1_000_000):
    output = Path(processed_data_dir)
    output.mkdir(parents=True, exist_ok=True)
    totals = None
    first, last, rows = None, None, 0
    for chunk in pd.read_csv(Path(raw_data_dir) / "transactions_train.csv",
                             usecols=["t_dat", "article_id"], dtype={"article_id": str}, chunksize=chunksize):
        dates = pd.to_datetime(chunk.t_dat, errors="raise")
        if dates.isna().any() or chunk.article_id.isna().any():
            raise ValueError("Transactions contain missing date or article_id")
        chunk["article_id"] = chunk.article_id.str.zfill(10)
        chunk["year_week"] = normalize_year_week(dates)
        part = chunk.groupby(["article_id", "year_week"]).size()
        totals = part if totals is None else totals.add(part, fill_value=0)
        first = dates.min() if first is None else min(first, dates.min())
        last = dates.max() if last is None else max(last, dates.max())
        rows += len(chunk)
        print(f"Aggregated {rows:,} transactions", flush=True)
    if totals is None or rows == 0:
        raise ValueError("No transactions")
    weekly = totals.astype("int64").rename("demand").reset_index()
    calendar = week_calendar(first, last)
    weekly.to_csv(output / "weekly_sales_all.csv", index=False)
    calendar.to_csv(output / "week_calendar.csv", index=False)
    write_json(output / "aggregation_metadata.json", {
        "transaction_count": rows, "first_date": str(first.date()), "last_date": str(last.date()),
        "n_observed_calendar_weeks": len(calendar), "n_complete_weeks": int(calendar.is_complete_week.sum()),
        "target_definition": "observed transaction row count; not unconstrained latent demand",
    })
    return weekly


if __name__ == "__main__":
    build_weekly_sales()
