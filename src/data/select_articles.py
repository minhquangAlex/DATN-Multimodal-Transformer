from pathlib import Path
from dataclasses import asdict
import re
import numpy as np
import pandas as pd
from src.config import DataConfig, file_sha256, write_json
from src.data.splits import assign_time_split


def clean_text(value):
    if pd.isna(value):
        return ""
    text = str(value).strip()
    text = re.sub(r"\s+", " ", text)
    return text


def build_text_for_embedding(row):
    """Ghép text embedding theo nguyên tắc rõ ràng, có fallback khi thiếu trường."""
    prod_name = clean_text(row.get('prod_name'))
    product_type_name = clean_text(row.get('product_type_name'))
    product_group_name = clean_text(row.get('product_group_name'))
    detail_desc = clean_text(row.get('detail_desc'))

    if detail_desc:
        text = f"{prod_name}. {product_type_name} in {product_group_name}. {detail_desc}"
    else:
        text_parts = [p for p in [prod_name, product_type_name, product_group_name] if p]
        if text_parts:
            text = ". ".join(text_parts)
        else:
            text = "product description unavailable"

    # Loại bỏ các dấu chấm thừa
    text = text.replace("..", ".").strip()
    return text


def stratified_sample(frame, size, seed=42):
    """Exact proportional allocation, preserving every group when budget permits."""
    if frame.empty or size < 1:
        raise ValueError("No eligible products or invalid sample size")
    frame = frame.sort_values('article_id').copy()
    frame['product_group_name'] = frame.product_group_name.fillna('Unknown')
    counts = frame.groupby('product_group_name', sort=True).size()
    size = min(size, len(frame))
    allocation = pd.Series(0, index=counts.index, dtype=int)
    if size >= len(counts):
        allocation[:] = 1
    capacity = counts - allocation
    remaining = size - int(allocation.sum())
    if remaining:
        quotas = capacity / capacity.sum() * remaining
        extra = np.floor(quotas).astype(int)
        allocation += extra
        residual = remaining - int(extra.sum())
        for group in (quotas - extra).sort_values(ascending=False, kind='stable').index[:residual]:
            allocation[group] += 1
    return pd.concat([
        group.sample(n=int(allocation[name]), random_state=seed)
        for name, group in frame.groupby('product_group_name', sort=True)
        if allocation[name] > 0
    ]).sort_values('article_id').reset_index(drop=True)


def select_articles(raw_data_dir="raw_data", processed_data_dir="processed_data/v2", config=None):
    config = config or DataConfig()
    output = Path(processed_data_dir)
    articles = pd.read_csv(Path(raw_data_dir) / 'articles.csv', dtype={'article_id': str})
    articles['article_id'] = articles.article_id.str.zfill(10)
    if articles.article_id.isna().any() or articles.article_id.duplicated().any():
        raise ValueError('Article IDs must be present and unique')
    weekly = pd.read_csv(output / 'weekly_sales_all.csv', dtype={'article_id': str})
    calendar = pd.read_csv(output / 'week_calendar.csv')
    calendar = assign_time_split(calendar.loc[calendar.is_complete_week.eq(True)], config.train_end, config.val_end)
    if set(calendar.split) != {'train', 'val', 'test'}:
        raise ValueError('The configured dates must produce nonempty train/val/test periods')
    weekly = weekly[weekly.year_week.isin(calendar.year_week)]
    train = weekly[weekly.year_week.isin(calendar.loc[calendar.split.eq('train'), 'year_week'])]
    stats = train.loc[train.demand.gt(0)].groupby('article_id').agg(
        active_weeks=('year_week', 'nunique'), total_transactions=('demand', 'sum')).reset_index()
    candidates = stats.loc[(stats.active_weeks >= config.min_active_weeks) &
                           (stats.total_transactions >= config.min_total_transactions)]
    candidates = candidates.merge(articles, on='article_id', validate='one_to_one')
    selected = stratified_sample(candidates, config.target_sample_size, config.seed)
    selected['detail_desc'] = selected.detail_desc.fillna('').map(clean_text)
    selected['text_available'] = selected.apply(
        lambda row: any(clean_text(row.get(c)) for c in ['prod_name', 'product_type_name', 'product_group_name', 'detail_desc']), axis=1)
    selected['text_for_embedding'] = selected.apply(build_text_for_embedding, axis=1)
    ids = selected.article_id.tolist()
    grid = pd.MultiIndex.from_product([ids, calendar.year_week], names=['article_id', 'year_week']).to_frame(index=False)
    weekly_selected = weekly.loc[weekly.article_id.isin(ids)].copy()
    index = grid.merge(weekly_selected, on=['article_id', 'year_week'], how='left', validate='one_to_one')
    index['demand'] = index.demand.fillna(0).astype('int32')
    index = index.merge(calendar, on='year_week', validate='many_to_one')
    selected.to_csv(output / 'selected_articles.csv', index=False)
    index.to_csv(output / 'dataset_index.csv', index=False, float_format='%.8f')
    weekly_selected.to_csv(output / 'weekly_sales_selected.csv', index=False)
    summary = {
        'schema_version': 2, 'config': asdict(config), 'selection_period': 'train only',
        'n_candidates': len(candidates), 'n_selected_articles': len(selected),
        'n_dataset_rows': len(index), 'n_weeks': len(calendar),
        'zero_fraction': float(index.demand.eq(0).mean()),
        'n_missing_detail_desc': int(selected.detail_desc.eq('').sum()),
        'group_distribution': selected.product_group_name.value_counts().to_dict(),
        'splits': {name: {'first_week': rows.year_week.min(), 'last_week': rows.year_week.max(), 'n_weeks': len(rows)}
                   for name, rows in calendar.groupby('split')},
        'selected_articles_sha256': file_sha256(output / 'selected_articles.csv'),
        'dataset_index_sha256': file_sha256(output / 'dataset_index.csv'),
        'boundary_policy': 'exclude incomplete first and last calendar weeks',
    }
    write_json(output / 'stage1_summary.json', summary)
    print(f"Selected {len(selected):,} products; {len(calendar)} complete weeks; candidates from train only", flush=True)
    return summary


if __name__ == '__main__':
    select_articles()
