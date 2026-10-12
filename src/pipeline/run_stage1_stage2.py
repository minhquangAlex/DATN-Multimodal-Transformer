"""Build stage 1. Stage 2 extraction is a separate Kaggle job."""
import argparse
from src.config import DataConfig
from src.data.build_weekly_sales import build_weekly_sales
from src.data.select_articles import select_articles
from src.data.splits import assign_time_split


def run_stage1_stage2(raw_data_dir='raw_data', processed_data_dir='processed_data/v2', config=None, reuse_aggregation=False):
    if not reuse_aggregation:
        build_weekly_sales(raw_data_dir, processed_data_dir)
    summary = select_articles(raw_data_dir, processed_data_dir, config or DataConfig())
    print('Stage 1 complete. Extract new embeddings from this exact selected_articles.csv on Kaggle.')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw-data-dir', default='raw_data')
    parser.add_argument('--processed-data-dir', default='processed_data/v2')
    parser.add_argument('--reuse-aggregation', action='store_true', help='Reuse the unfiltered weekly_sales_all.csv and calendar')
    args = parser.parse_args()
    run_stage1_stage2(**vars(args))
