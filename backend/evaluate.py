"""Report model provenance, size, and held-out next-byte prediction loss."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ai_core.model import IndependentLanguageModel, ModelConfig  # noqa: E402
from tokenizer.byte_tokenizer import ByteTokenizer  # noqa: E402


def main() -> None:
    checkpoint_path = ROOT / "checkpoints" / "independent_lm_v0_1.pt"
    if not checkpoint_path.exists():
        raise SystemExit("No checkpoint yet. Run: python training/train.py")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ModelConfig(**checkpoint["model_config"])
    model = IndependentLanguageModel(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    corpus = (ROOT / "data" / "seed_corpus.txt").read_text(encoding="utf-8")
    tokens = torch.tensor(ByteTokenizer().encode(corpus), dtype=torch.long)
    validation = tokens[int(0.9 * len(tokens)) :]
    context = config.context_length
    with torch.no_grad():
        x = validation[:-1].unfold(0, context, context)
        y = validation[1:].unfold(0, context, context)
        losses = []
        for start in range(0, len(x), 16):
            _, loss = model(x[start : start + 16].to(device), y[start : start + 16].to(device))
            losses.append(loss.item())
    val_loss = sum(losses) / max(1, len(losses))
    report = {
        "project": "Independent AI Lab v0.1",
        "parameters": model.parameter_count(),
        "architecture": config.__dict__,
        "validation_next_byte_loss": round(val_loss, 4),
        "validation_perplexity": round(float(torch.exp(torch.tensor(val_loss))), 3),
        "training_provenance": checkpoint["training"],
        "limitations": [
            "This is a small byte-level language-model experiment, not a ChatGPT-equivalent assistant.",
            "It has not been trained for image, audio, or video generation.",
            "Its starter corpus is small and synthetic/original; its knowledge and reliability are limited.",
        ],
    }
    out = ROOT / "evaluation" / "latest_report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
