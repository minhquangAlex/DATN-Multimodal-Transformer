import json
import numpy as np
import pandas as pd
import pytest
from src.config import file_sha256
from src.data.dataset import HMDataModule
from src.data.select_articles import select_articles, stratified_sample
from src.data.splits import assign_time_split
from src.data.build_weekly_sales import week_calendar


def test_dates_and_iso_boundary():
    frame = pd.DataFrame({'year_week': ['2019-W52', '2020-W01', '2020-W02']})
    result = assign_time_split(frame, '2019-12-29', '2020-01-05')
    assert result.split.tolist() == ['train', 'val', 'test']
    with pytest.raises(ValueError):
        assign_time_split(frame, '2020-01-01', '2020-01-05')


def test_complete_week_policy():
    calendar = week_calendar('2018-09-20', '2020-09-22')
    assert len(calendar) == 106
    assert calendar.is_complete_week.sum() == 104
    assert not calendar.iloc[0].is_complete_week
    assert not calendar.iloc[-1].is_complete_week


def test_selection_ignores_future_and_does_not_shrink_source(stage1_data):
    raw, processed, config = stage1_data
    original = file_sha256(processed/'weekly_sales_all.csv')
    selected = pd.read_csv(processed/'selected_articles.csv', dtype={'article_id': str})
    assert set(selected.article_id) == {'0000000001', '0000000002'}
    assert selected.active_weeks.tolist() == [6, 6]
    assert selected.total_transactions.tolist() == [21, 21]
    first = file_sha256(processed/'selected_articles.csv')
    select_articles(raw, processed, config)
    assert file_sha256(processed/'weekly_sales_all.csv') == original
    assert file_sha256(processed/'selected_articles.csv') == first


def test_sampling_exact_and_preserves_small_groups():
    frame = pd.DataFrame({'article_id': [str(i) for i in range(100)],
                          'product_group_name': ['big']*98+['small', 'rare']})
    chosen = stratified_sample(frame, 10)
    assert len(chosen) == 10
    assert chosen.product_group_name.nunique() == 3
    assert len(stratified_sample(frame, 1000)) == 100
    with pytest.raises(ValueError):
        stratified_sample(frame.iloc[:0], 10)


def test_windows_respect_targets_and_keep_past_context(stage1_data):
    _, processed, _ = stage1_data
    dm = HMDataModule(processed/'dataset_index.csv', seq_len=3, pred_len=2, use_embeddings=False)
    dm.setup()
    assert [len(getattr(dm, name+'_dataset')) for name in ['train', 'val', 'test']] == [4, 4, 4]
    for name in ['train', 'val', 'test']:
        dataset = getattr(dm, name+'_dataset')
        calendar = pd.read_csv(processed/'dataset_index.csv').drop_duplicates('year_week').set_index('year_week')
        for origin in dataset.origins:
            assert set(calendar.loc[dataset.weeks[origin:origin+2], 'split']) == {name}
    assert dm.val_dataset[0]['forecast_origin'] == '2020-W08'
    assert dm.val_dataset[0]['history'].flatten().tolist() == [4., 5., 6.]
    assert dm.val_dataset[0]['target'].tolist() == [7., 8.]
    assert dm.train_dataset.sales is dm.test_dataset.sales


def test_tampered_index_and_empty_windows_fail(stage1_data):
    _, processed, _ = stage1_data
    with pytest.raises(ValueError, match='No valid'):
        HMDataModule(processed/'dataset_index.csv', seq_len=30, use_embeddings=False).setup()
    frame = pd.read_csv(processed/'dataset_index.csv', dtype={'article_id': str})
    frame.loc[0, 'demand'] = 999
    frame.to_csv(processed/'dataset_index.csv', index=False)
    with pytest.raises(ValueError, match='hash mismatch'):
        HMDataModule(processed/'dataset_index.csv', seq_len=3, use_embeddings=False).setup()


def test_chunked_eda_and_input_ambiguity(stage1_data, tmp_path):
    from src.data.eda import run_eda
    from src.utils.paths import unique_directory
    raw, _, _ = stage1_data
    report = run_eda(raw, tmp_path/'eda', chunksize=17)
    assert report['n_transactions'] == 457
    assert report['missing_detail_desc'] == 1
    assert (tmp_path/'eda/daily_sales.csv').exists()
    with pytest.raises(ValueError, match='exactly one'):
        unique_directory([tmp_path/'old', tmp_path/'new'], 'Stage 1')
