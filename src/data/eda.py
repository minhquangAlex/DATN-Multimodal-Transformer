"""Chunked exploratory summaries; no full 3.5 GB transaction frame in RAM."""
from pathlib import Path
import pandas as pd
from src.config import write_json


def run_eda(raw_data_dir='raw_data', output_dir=None, chunksize=1_000_000):
    raw = Path(raw_data_dir)
    articles = pd.read_csv(raw/'articles.csv', dtype={'article_id': str})
    descriptions = articles.detail_desc.fillna('').astype(str).str.strip()
    counts, daily = None, None
    total = 0
    for chunk in pd.read_csv(raw/'transactions_train.csv', usecols=['article_id', 't_dat'],
                             dtype={'article_id': str}, chunksize=chunksize):
        dates = pd.to_datetime(chunk.t_dat, errors='raise')
        item_counts = chunk.article_id.value_counts()
        day_counts = dates.value_counts()
        counts = item_counts if counts is None else counts.add(item_counts, fill_value=0)
        daily = day_counts if daily is None else daily.add(day_counts, fill_value=0)
        total += len(chunk)
    if total == 0:
        raise ValueError('Empty transactions')
    summary = {'n_articles': len(articles), 'unique_article_ids': articles.article_id.nunique(),
               'missing_detail_desc': int(descriptions.eq('').sum()),
               'missing_detail_desc_fraction': float(descriptions.eq('').mean()),
               'n_transactions': total, 'n_transacted_articles': len(counts),
               'first_date': str(daily.index.min().date()), 'last_date': str(daily.index.max().date()),
               'transactions_per_article': counts.describe().to_dict(),
               'product_group_distribution': articles.product_group_name.value_counts().to_dict(),
               'target_limitation': 'Observed sales count; inventory, availability and unmet demand are not observed.'}
    if output_dir:
        output = Path(output_dir)
        write_json(output/'eda_summary.json', summary)
        daily.sort_index().astype('int64').rename_axis('date').rename('sales').to_csv(output/'daily_sales.csv')
        counts.astype('int64').rename_axis('article_id').rename('transactions').to_csv(output/'transactions_per_article.csv')
    print(summary)
    return summary


if __name__ == '__main__':
    run_eda(output_dir='results/eda/v2')
