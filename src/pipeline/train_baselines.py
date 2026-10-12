"""Reproducible baseline and same-checkpoint robustness experiments."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import platform
from pathlib import Path
import time
import uuid
import numpy as np
import pandas as pd
import torch
from src.config import file_sha256, write_json
from src.data.dataset import HMDataModule
from src.models.baselines import (ZeroBaseline, LastValueNaiveBaseline, MovingAverageBaseline,
                                 LinearRegressionBaseline, MLPBaseline, SimpleMultimodalBaseline)
from src.models.rnn_sales import LSTMSalesModel
from src.models.sales_transformer import SalesTransformer
from src.utils.metrics import evaluate_all_metrics, metric_table
from src.utils.reproducibility import seed_everything
from src.utils.trainer import BaseTrainer, prediction_frame


MODEL_NAMES = ['zero', 'last_value', 'moving_average', 'linear', 'mlp', 'lstm', 'transformer', 'multimodal_mlp']


def non_learning_predictions(model, loader):
    truths, predictions, ids, origins = [], [], [], []
    for batch in loader:
        predictions.append(np.maximum(model.predict(batch['history']), 0))
        truths.append(batch['target'].numpy())
        ids.extend(batch['article_id'])
        origins.extend(batch['forecast_origin'])
    if not truths:
        raise ValueError('Cannot evaluate empty loader')
    return prediction_frame(np.concatenate(truths), np.concatenate(predictions), ids, origins)


def evaluate_non_learning_baseline(baseline_model, test_loader):
    frame = non_learning_predictions(baseline_model, test_loader)
    return evaluate_all_metrics(frame.actual.to_numpy(), frame.prediction.to_numpy())


def run_all_baselines(epochs=30, batch_size=256, patience=5, seeds=(42, 43, 44),
                      processed_data_dir='processed_data/v2', embeddings_dir='embeddings/v2',
                      output_dir='results/v2', models=None, seq_len=12, pred_len=2,
                      target_transform='log1p', retrained_ablation=False, num_workers=0, device=None):
    models = list(MODEL_NAMES if models is None else models)
    seeds = list(seeds)
    if not models or len(set(models)) != len(models) or set(models)-set(MODEL_NAMES):
        raise ValueError(f'Choose unique models from {MODEL_NAMES}')
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('Provide distinct seeds')
    if epochs < 1 or patience < 1:
        raise ValueError('epochs and patience must be positive')
    if retrained_ablation and 'multimodal_mlp' not in models:
        raise ValueError('Retrained ablation requires multimodal_mlp')
    processed, embeddings = Path(processed_data_dir), Path(embeddings_dir)
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid.uuid4().hex[:8]
    output = Path(output_dir) / run_id
    dm = HMDataModule(dataset_index_path=processed/'dataset_index.csv',
                      image_emb_path=embeddings/'image/image_embeddings.pt',
                      text_emb_path=embeddings/'text/text_embeddings.pt', metadata_path=embeddings/'metadata.json',
                      seq_len=seq_len, pred_len=pred_len, batch_size=batch_size, num_workers=num_workers,
                      use_embeddings='multimodal_mlp' in models, require_full=False)
    # Validate before creating a run or starting any expensive training.
    dm.setup()
    selected = pd.read_csv(processed/'selected_articles.csv', dtype={'article_id': str})
    metadata = selected[['article_id', 'product_group_name']].copy()
    sales = dm.train_dataset.sales
    first_val = dm.val_dataset.origins[0]
    sparsity = (sales[:, :first_val] == 0).mean(axis=1)
    sparse = pd.DataFrame({'article_id': dm.train_dataset.article_ids, 'train_zero_fraction': sparsity})
    sparse['sparsity_group'] = np.where(sparsity < .5, 'zero_fraction<0.5', 'zero_fraction>=0.5')
    metadata = metadata.merge(sparse, on='article_id', validate='one_to_one')
    manifest = {
        'status': 'running', 'run_id': run_id, 'models': models, 'seeds': seeds,
        'epochs': epochs, 'patience': patience, 'seq_len': seq_len, 'pred_len': pred_len,
        'batch_size': batch_size, 'num_workers': num_workers, 'target_transform': target_transform,
        'retrained_ablation': retrained_ablation,
        'device': device or ('cuda' if torch.cuda.is_available() else 'cpu'), 'python': platform.python_version(),
        'packages': {name: importlib.metadata.version(name) for name in ['torch', 'numpy', 'pandas']},
        'data_hashes': {name: file_sha256(processed/name) for name in ['selected_articles.csv', 'dataset_index.csv', 'stage1_summary.json']},
        'source_hashes': {str(p): file_sha256(p) for p in sorted(Path(__file__).resolve().parents[1].rglob('*.py'))},
        'split_summary': dm.split_summary, 'embedding_report': dm.embedding_report,
        'metric_policy': 'original sales units; wMAPE and sMAPE in percent; undefined wMAPE stored blank',
        'forecast_protocol': 'rolling origin, all H targets inside the same split; observed past test weeks may enter later history',
        'forecast_origin_definition': 'first forecast week, immediately after the observed history',
    }
    if dm.use_embeddings:
        manifest['embedding_metadata_sha256'] = file_sha256(embeddings/'metadata.json')
    write_json(output/'run_config.json', manifest)
    rows, detailed = [], []

    def record(frame, name, seed, protocol='baseline', scenario='full'):
        frame = frame.merge(metadata, on='article_id', validate='many_to_one')
        tags = {'Model': name, 'seed': seed, 'protocol': protocol, 'scenario': scenario}
        for key, value in tags.items():
            frame[key] = value
        prediction_dir = output/'predictions'
        prediction_dir.mkdir(exist_ok=True)
        frame.to_csv(prediction_dir/f'{name}_seed{seed}_{protocol}_{scenario}.csv.gz', index=False)
        result = evaluate_all_metrics(frame.actual.to_numpy(), frame.prediction.to_numpy())
        rows.append({**tags, **result})
        for group in ['horizon', 'product_group_name', 'sparsity_group']:
            table = metric_table(frame, [group])
            table = table.rename(columns={group: 'group_value'})
            table['group_type'] = group
            for key, value in tags.items():
                table[key] = value
            detailed.append(table)
        pd.DataFrame(rows).to_csv(output/'baseline_results.csv', index=False)
        pd.concat(detailed, ignore_index=True).to_csv(output/'detailed_metrics.csv', index=False)
        print({**tags, **result}, flush=True)
        return result

    factories = {
        'linear': lambda: LinearRegressionBaseline(seq_len, pred_len),
        'mlp': lambda: MLPBaseline(seq_len, pred_len),
        'lstm': lambda: LSTMSalesModel(pred_len=pred_len),
        'transformer': lambda: SalesTransformer(pred_len=pred_len),
        'multimodal_mlp': lambda: SimpleMultimodalBaseline(seq_len, pred_len,
            image_dim=dm.train_dataset.image_dim, text_dim=dm.train_dataset.text_dim),
    }
    naive_models = {'zero': ZeroBaseline(pred_len), 'last_value': LastValueNaiveBaseline(pred_len),
                    'moving_average': MovingAverageBaseline(pred_len, min(4, seq_len))}
    scenarios = {
        'full': {}, 'missing_image': {'image': False}, 'missing_text': {'text': False},
        'partial_history': {'history_keep_weeks': max(1, seq_len//2)},
        'missing_image_and_text': {'image': False, 'text': False},
    }
    started = time.perf_counter()
    try:
        for name in models:
            if name in naive_models:
                record(non_learning_predictions(naive_models[name], dm.test_dataloader()), name, -1)
                continue
            for seed in seeds:
                seed_everything(seed)
                dm.seed = seed
                model = factories[name]()
                trainer = BaseTrainer(model, device=device, model_name=name,
                                      checkpoint_dir=output/'checkpoints'/f'seed{seed}', target_transform=target_transform)
                trainer.fit(dm.train_dataloader(), dm.val_dataloader(), epochs, patience)
                full_frame = trainer.prediction_frame(dm.test_dataloader())
                record(full_frame, name, seed)
                if name == 'multimodal_mlp':
                    # Exactly the same weights are used for every missing-input scenario.
                    record(full_frame, name, seed, 'same_checkpoint', 'full')
                    for scenario, config in scenarios.items():
                        if scenario == 'full':
                            continue
                        frame = trainer.prediction_frame(dm.test_dataloader(), config)
                        record(frame, name, seed, 'same_checkpoint', scenario)
                    if retrained_ablation:
                        record(full_frame, name, seed, 'retrained', 'full')
                        for scenario in ['missing_image', 'missing_text', 'missing_image_and_text']:
                            seed_everything(seed)
                            ablation = BaseTrainer(factories[name](), device=device, model_name=f'{name}_{scenario}',
                                checkpoint_dir=output/'checkpoints'/f'seed{seed}', target_transform=target_transform,
                                modality_config=scenarios[scenario])
                            ablation.fit(dm.train_dataloader(), dm.val_dataloader(), epochs, patience)
                            record(ablation.prediction_frame(dm.test_dataloader()), name, seed, 'retrained', scenario)
        results = pd.DataFrame(rows)
        metric_names = ['MAE', 'RMSE', 'wMAPE (%)', 'sMAPE (%)']
        summary = results.groupby(['Model', 'protocol', 'scenario'])[metric_names].agg(['mean', 'std', 'count'])
        summary.columns = ['_'.join(c) for c in summary.columns]
        summary.reset_index().to_csv(output/'seed_summary.csv', index=False)
        robustness = results[results.protocol.eq('same_checkpoint')].copy()
        if not robustness.empty:
            full = robustness[robustness.scenario.eq('full')][['Model', 'seed', 'MAE']].rename(columns={'MAE': 'full_MAE'})
            robustness = robustness.merge(full, on=['Model', 'seed'], validate='many_to_one')
            robustness['MAE_increase_pct'] = np.where(robustness.full_MAE > 0,
                100*(robustness.MAE-robustness.full_MAE)/robustness.full_MAE, np.nan)
            robustness.to_csv(output/'robustness_comparison.csv', index=False)
        manifest['status'] = 'complete'
        return output
    except Exception as exc:
        manifest.update(status='failed', error=str(exc))
        raise
    finally:
        manifest['elapsed_seconds'] = time.perf_counter()-started
        write_json(output/'run_config.json', manifest)
        print(f'Results: {output}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    parser.add_argument('--models', nargs='+', choices=MODEL_NAMES, default=None)
    parser.add_argument('--processed-data-dir', default='processed_data/v2')
    parser.add_argument('--embeddings-dir', default='embeddings/v2')
    parser.add_argument('--output-dir', default='results/v2')
    parser.add_argument('--seq-len', type=int, default=12)
    parser.add_argument('--pred-len', type=int, default=2)
    parser.add_argument('--target-transform', choices=['none', 'log1p'], default='log1p')
    parser.add_argument('--retrained-ablation', action='store_true')
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--device', default=None)
    run_all_baselines(**vars(parser.parse_args()))
