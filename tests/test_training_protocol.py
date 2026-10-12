import json
import numpy as np
import pandas as pd
import pytest
import torch
from src.models.baselines import LastValueNaiveBaseline, SeasonalNaiveBaseline, LinearRegressionBaseline
from src.pipeline.train_baselines import run_all_baselines
from src.utils.metrics import evaluate_all_metrics
from src.utils.trainer import BaseTrainer


def test_naive_and_seasonal_are_different():
    history = torch.arange(1., 53.).reshape(1,52,1)
    np.testing.assert_equal(LastValueNaiveBaseline(2).predict(history), [[52,52]])
    np.testing.assert_equal(SeasonalNaiveBaseline(2).predict(history), [[1,2]])
    with pytest.raises(ValueError, match='52'):
        SeasonalNaiveBaseline().predict(history[:,:12])


def test_metrics_zero_actual_and_shape_checks():
    result = evaluate_all_metrics(np.array([0.,2.]), np.array([0.,1.]))
    assert result['MAE'] == .5
    assert result['wMAPE (%)'] == 50.
    assert result['sMAPE (%)'] == pytest.approx(100/3)
    result = evaluate_all_metrics(np.zeros(2), np.zeros(2))
    assert np.isnan(result['wMAPE (%)'])
    assert result['sMAPE (%)'] == 0
    with pytest.raises(ValueError, match='shape'):
        evaluate_all_metrics(np.zeros((2,1)), np.zeros(2))


def test_transform_inverse_and_nonfinite(tmp_path):
    trainer = BaseTrainer(LinearRegressionBaseline(3,2), device='cpu', checkpoint_dir=tmp_path)
    values = torch.tensor([0., 3., 100.])
    torch.testing.assert_close(trainer._inverse(torch.log1p(values)), values)
    with pytest.raises(ValueError, match='Nonfinite'):
        trainer._inverse(torch.tensor([1000.]))


def test_full_pipeline_saves_checkpoints_and_both_protocols(embedding_bundle, tmp_path):
    processed, embeddings = embedding_bundle
    output = run_all_baselines(epochs=1, patience=1, seeds=[7], batch_size=8, seq_len=3, pred_len=2,
        processed_data_dir=processed, embeddings_dir=embeddings, output_dir=tmp_path/'results',
        models=['zero','linear','multimodal_mlp'], retrained_ablation=True, device='cpu')
    config = json.loads((output/'run_config.json').read_text())
    assert config['status'] == 'complete'
    result = pd.read_csv(output/'baseline_results.csv')
    assert len(result[result.protocol.eq('same_checkpoint')]) == 5
    assert len(result[result.protocol.eq('retrained')]) == 4
    full = result[(result.Model=='multimodal_mlp') & (result.scenario=='full')]
    assert full.MAE.nunique() == 1
    predictions = pd.read_csv(next((output/'predictions').glob('linear*.csv.gz')))
    assert {'article_id','forecast_origin','target_week','horizon','actual','prediction','product_group_name'}.issubset(predictions.columns)
    assert len(predictions) == 8
    assert predictions.prediction.ge(0).all()
    assert (output/'robustness_comparison.csv').exists()
    checkpoint = torch.load(output/'checkpoints/seed7/best_linear.pt', weights_only=True)
    assert checkpoint['epoch'] == 1
    assert checkpoint['target_transform'] == 'log1p'


def test_sales_only_learning_architectures_without_embeddings(stage1_data, tmp_path):
    _, processed, _ = stage1_data
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        output = run_all_baselines(epochs=1, patience=1, seeds=[13], batch_size=8, seq_len=3, pred_len=2,
            processed_data_dir=processed, embeddings_dir=tmp_path/'does-not-exist', output_dir=tmp_path/'runs',
            models=['mlp', 'lstm', 'transformer'], device='cpu')
    finally:
        torch.set_num_threads(previous_threads)
    result = pd.read_csv(output/'baseline_results.csv')
    assert set(result.Model) == {'mlp', 'lstm', 'transformer'}
    assert result.MAE.notna().all()
    assert json.loads((output/'run_config.json').read_text())['embedding_report'] is None
