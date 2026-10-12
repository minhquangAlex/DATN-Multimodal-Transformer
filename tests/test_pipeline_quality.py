import pandas as pd

from src.data.build_weekly_sales import normalize_year_week
from src.data.select_articles import build_text_for_embedding


def test_normalize_year_week_returns_iso_format():
    s = pd.to_datetime(["2018-09-20", "2020-09-22"])
    out = normalize_year_week(s)
    assert len(out) == 2
    assert all(isinstance(x, str) for x in out)
    assert all(x.count("-W") == 1 for x in out)
    assert all(len(x.split("-W")[1]) == 2 for x in out)


def test_build_text_for_embedding_uses_fallback_when_detail_desc_missing():
    row = {
        "prod_name": "T-shirt",
        "product_type_name": "Shirt",
        "product_group_name": "Garment",
        "detail_desc": None,
    }

    text = build_text_for_embedding(row)

    assert "T-shirt" in text
    assert "Shirt" in text
    assert "Garment" in text
