from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ai_core.model import IndependentLanguageModel, ModelConfig
from tokenizer.byte_tokenizer import ByteTokenizer


def test_byte_tokenizer_round_trip():
    tokenizer = ByteTokenizer()
    sample = "Hello, 世界!"
    assert tokenizer.decode(tokenizer.encode(sample)) == sample


def test_model_forward_and_loss():
    config = ModelConfig(context_length=16, d_model=32, n_heads=4, n_layers=1)
    model = IndependentLanguageModel(config)
    x = torch.randint(0, 256, (2, 16))
    logits, loss = model(x, x)
    assert logits.shape == (2, 16, 256)
    assert loss is not None and torch.isfinite(loss)


def test_autoregressive_generation_shape():
    config = ModelConfig(context_length=16, d_model=32, n_heads=4, n_layers=1)
    model = IndependentLanguageModel(config)
    prompt = torch.randint(0, 256, (1, 5))
    output = model.generate(prompt, max_new_tokens=3, top_k=10)
    assert output.shape == (1, 8)
