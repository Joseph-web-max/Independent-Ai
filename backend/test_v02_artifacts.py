from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ai_core.model import IndependentLanguageModel, ModelConfig
from tokenizer.byte_tokenizer import ByteTokenizer
from training.data import load_document_splits


def test_v02_checkpoint_is_from_random_initialization_and_has_expected_shape():
    checkpoint_path = ROOT / "checkpoints" / "independent_lm_v0_2.pt"
    assert checkpoint_path.exists()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    training = checkpoint["training"]
    assert training["random_initialization"] is True
    assert training["v0_1_weights_loaded"] is False
    assert training["third_party_pretrained_weights_used"] is False
    assert training["external_ai_apis_used"] is False
    config = ModelConfig(**checkpoint["model_config"])
    assert config.context_length == 192
    assert config.d_model == 96
    assert config.n_layers == 3
    model = IndependentLanguageModel(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    assert model.parameter_count() == 377568


def test_v02_document_split_is_reproducible_and_nonempty():
    train_a, val_a = load_document_splits(seed=20261119)
    train_b, val_b = load_document_splits(seed=20261119)
    assert train_a == train_b
    assert val_a == val_b
    assert len(train_a) > len(val_a) > 1000


def test_v02_checkpoint_can_generate_tokens():
    checkpoint = torch.load(ROOT / "checkpoints" / "independent_lm_v0_2.pt", map_location="cpu", weights_only=False)
    config = ModelConfig(**checkpoint["model_config"])
    model = IndependentLanguageModel(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    prompt = torch.tensor([ByteTokenizer().encode("USER: Explain learning.\nASSISTANT:")], dtype=torch.long)
    output = model.generate(prompt, max_new_tokens=4, temperature=0.8, top_k=24)
    assert output.shape[1] == prompt.shape[1] + 4
