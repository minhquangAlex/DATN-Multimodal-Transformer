import json
import pandas as pd
import pytest
import torch
from src.config import file_sha256, write_json
from src.models.baselines import SimpleMultimodalBaseline
from src.utils.trainer import BaseTrainer
from src.features.verify_embeddings import validate_embedding_dict, load_embedding_bundle
from src.features.extract_embeddings import audit_images
from PIL import Image


def test_masks_change_inputs_without_mutating_batch(tmp_path):
    model = SimpleMultimodalBaseline(seq_len=4, pred_len=2, image_dim=3, text_dim=2)
    trainer = BaseTrainer(model, device='cpu', checkpoint_dir=tmp_path, target_transform='none')
    batch = {'history': torch.arange(8.).reshape(2,4,1), 'target': torch.ones(2,2),
             'image_emb': torch.ones(2,3), 'text_emb': torch.ones(2,2),
             'image_available': torch.tensor([True, False]), 'text_available': torch.tensor([True, True])}
    original_weights = {k: v.clone() for k,v in model.state_dict().items()}
    inputs, _ = trainer._prepare_batch(batch, {'image': False, 'history_keep_weeks': 2})
    assert inputs['image_emb'].count_nonzero() == 0
    assert inputs['text_emb'].count_nonzero() == 4
    assert inputs['history'][:, :2].count_nonzero() == 0
    assert inputs['history_mask'][:, :2].sum() == 0
    assert torch.equal(inputs['history'][:, 2:], batch['history'][:, 2:])
    assert batch['image_emb'].sum() == 6
    trainer.evaluate([batch], {'text': False})
    assert all(torch.equal(value, model.state_dict()[key]) for key,value in original_weights.items())
    assert trainer.modality_config['text'] is True


def test_collapsed_and_bad_late_embedding_rejected():
    with pytest.raises(ValueError, match='collapsed'):
        validate_embedding_dict('image', {'0000000001': torch.ones(3), '0000000002': torch.ones(3)}, 3)
    with pytest.raises(ValueError, match='shape'):
        validate_embedding_dict('image', {'0000000001': torch.ones(3), '0000000002': torch.ones(2)}, 3)


def test_bundle_provenance_and_masks(embedding_bundle):
    processed, folder = embedding_bundle
    def load(full=True):
        return load_embedding_bundle(processed/'selected_articles.csv', folder/'image/image_embeddings.pt',
                                     folder/'text/text_embeddings.pt', folder/'metadata.json', full)
    assert load()[3]['image']['distinct_vectors'] == 2
    availability = pd.read_csv(folder/'availability.csv', dtype={'article_id': str})
    key = availability.loc[0, 'article_id']
    availability.loc[0, 'image_available'] = False
    availability.to_csv(folder/'availability.csv', index=False)
    vectors = torch.load(folder/'image/image_embeddings.pt', weights_only=True)
    vectors[key] = torch.zeros(3)
    torch.save(vectors, folder/'image/image_embeddings.pt')
    metadata = json.loads((folder/'metadata.json').read_text())
    metadata['availability_sha256'] = file_sha256(folder/'availability.csv')
    metadata['image_sha256'] = file_sha256(folder/'image/image_embeddings.pt')
    write_json(folder/'metadata.json', metadata)
    with pytest.raises(ValueError, match='full-modality'):
        load()
    assert load(False)[3]['image']['available_count'] == 1
    metadata['selected_articles_sha256'] = 'stale'
    write_json(folder/'metadata.json', metadata)
    with pytest.raises(ValueError, match='provenance'):
        load(False)
    from src.features.verify_embeddings import verify_stage2_embeddings
    with pytest.raises(ValueError, match='provenance'):
        verify_stage2_embeddings(processed/'selected_articles.csv', folder/'image/image_embeddings.pt',
                                 folder/'text/text_embeddings.pt', folder/'metadata.json')
    assert json.loads((folder/'metadata.json').read_text())['validation'] == 'failed'
    assert json.loads((folder/'validation_report.json').read_text())['validation'] == 'failed'


def test_image_audit_reports_real_missing_and_corrupt(tmp_path):
    folder = tmp_path/'000'
    folder.mkdir()
    Image.new('RGB', (20,20), 'red').save(folder/'0000000001.jpg')
    (folder/'0000000003.jpg').write_bytes(b'not an image')
    selected = pd.DataFrame({'article_id': ['0000000001','0000000002','0000000003'], 'text_available': [True]*3})
    audit = audit_images(selected, tmp_path)
    assert audit.image_status.tolist() == ['ok', 'missing', 'unreadable']
    assert audit.image_available.tolist() == [True, False, False]


def test_extraction_writes_verified_bundle_with_real_image_inputs(stage1_data, tmp_path, monkeypatch):
    import sys
    import types
    import numpy as np
    import src.features.extract_embeddings as extraction
    _, processed, _ = stage1_data
    images = tmp_path/'images'
    (images/'000').mkdir(parents=True)
    Image.new('RGB', (20,20), 'red').save(images/'000/0000000001.jpg')
    Image.new('RGB', (20,20), 'blue').save(images/'000/0000000002.jpg')

    class Vision(torch.nn.Module):
        config = types.SimpleNamespace(hidden_size=3, _commit_hash='offline-test')
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()
        def forward(self, pixel_values):
            return types.SimpleNamespace(pooler_output=pixel_values.mean(dim=(2,3)))

    class Processor:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()
        def __call__(self, images, **kwargs):
            pixels = torch.stack([torch.from_numpy(np.array(im).copy()).permute(2,0,1).float() for im in images])
            return {'pixel_values': pixels}

    class Language:
        def __init__(self, *args, **kwargs):
            pass
        def get_sentence_embedding_dimension(self):
            return 2
        def encode(self, texts, **kwargs):
            return torch.eye(len(texts), 2)
        def __getitem__(self, index):
            return types.SimpleNamespace(auto_model=types.SimpleNamespace(config=types.SimpleNamespace(_commit_hash='offline-test')))

    transformers = types.ModuleType('transformers')
    transformers.CLIPVisionModel = Vision
    transformers.CLIPImageProcessor = Processor
    sentences = types.ModuleType('sentence_transformers')
    sentences.SentenceTransformer = Language
    monkeypatch.setitem(sys.modules, 'transformers', transformers)
    monkeypatch.setitem(sys.modules, 'sentence_transformers', sentences)
    monkeypatch.setattr(extraction.importlib.metadata, 'version', lambda name: 'offline-test')
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    output = tmp_path/'extracted'
    report = extraction.extract_embeddings(processed/'selected_articles.csv', images, output, batch_size=1)
    assert report['image']['distinct_vectors'] == 2
    assert json.loads((output/'metadata.json').read_text())['validation'] == 'passed'
    assert json.loads((output/'validation_report.json').read_text())['validation'] == 'passed'
    vectors = torch.load(output/'image/image_embeddings.pt', weights_only=True)
    assert all(value.norm().item() == pytest.approx(1.) for value in vectors.values())
