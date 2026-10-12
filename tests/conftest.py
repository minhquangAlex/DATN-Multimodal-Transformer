from dataclasses import asdict
import pandas as pd
import pytest
import torch
from src.config import DataConfig, file_sha256, write_json
from src.data.build_weekly_sales import build_weekly_sales
from src.data.select_articles import select_articles


@pytest.fixture
def stage1_data(tmp_path):
    raw, processed = tmp_path/'raw', tmp_path/'processed'
    raw.mkdir()
    ids = ['0000000001', '0000000002', '0000000003']
    pd.DataFrame({'article_id': ids, 'prod_name': ['A', 'B', 'Future'],
                  'product_type_name': ['Shirt']*3, 'product_group_name': ['Clothes']*3,
                  'detail_desc': ['cotton', None, 'future item']}).to_csv(raw/'articles.csv', index=False)
    dates = pd.date_range('2020-01-06', periods=12, freq='W-MON')
    rows = []
    for i, date in enumerate(dates):
        for key in ids[:2]:
            rows.extend([{'article_id': key, 't_dat': str(date.date())}]*(i+1))
        if i >= 9:
            rows.extend([{'article_id': ids[2], 't_dat': str(date.date())}]*100)
    rows.append({'article_id': ids[0], 't_dat': '2020-03-29'})
    pd.DataFrame(rows).to_csv(raw/'transactions_train.csv', index=False)
    config = DataConfig(train_end='2020-02-16', val_end='2020-03-08', min_active_weeks=2,
                        min_total_transactions=2, target_sample_size=2)
    build_weekly_sales(raw, processed, chunksize=37)
    select_articles(raw, processed, config)
    return raw, processed, config


@pytest.fixture
def embedding_bundle(stage1_data, tmp_path):
    _, processed, _ = stage1_data
    folder = tmp_path/'embeddings'
    (folder/'image').mkdir(parents=True)
    (folder/'text').mkdir()
    ids = pd.read_csv(processed/'selected_articles.csv', dtype={'article_id': str}).article_id.tolist()
    image = {key: torch.tensor([i+1., 2., 3.]) for i, key in enumerate(ids)}
    text = {key: torch.tensor([i+1., 2.]) for i, key in enumerate(ids)}
    torch.save(image, folder/'image/image_embeddings.pt')
    torch.save(text, folder/'text/text_embeddings.pt')
    pd.DataFrame({'article_id': ids, 'image_available': True, 'text_available': True}).to_csv(folder/'availability.csv', index=False)
    metadata = {'schema_version': 2, 'image_dim': 3, 'text_dim': 2,
                'selected_articles_sha256': file_sha256(processed/'selected_articles.csv'),
                'image_sha256': file_sha256(folder/'image/image_embeddings.pt'),
                'text_sha256': file_sha256(folder/'text/text_embeddings.pt'),
                'availability_sha256': file_sha256(folder/'availability.csv')}
    write_json(folder/'metadata.json', metadata)
    return processed, folder
