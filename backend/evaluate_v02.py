"""Evaluate v0.2 on document-level held-out text."""
from __future__ import annotations
import json
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ai_core.model import IndependentLanguageModel, ModelConfig  # noqa: E402
from tokenizer.byte_tokenizer import ByteTokenizer  # noqa: E402
from training.data import load_document_splits  # noqa: E402


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path = ROOT / "checkpoints" / "independent_lm_v0_2.pt"
    if not path.exists():
        raise SystemExit("Missing v0.2 checkpoint. Run python training/train_v02.py first.")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = ModelConfig(**checkpoint["model_config"])
    model = IndependentLanguageModel(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    _, val_text = load_document_splits(seed=checkpoint["training"]["seed"])
    ids = torch.tensor(ByteTokenizer().encode(val_text), dtype=torch.long)
    context = config.context_length
    starts = list(range(0, max(1, ids.numel() - context - 1), context))
    losses = []
    with torch.inference_mode():
        for start in starts:
            x = ids[start:start + context].unsqueeze(0).to(device)
            y = ids[start + 1:start + context + 1].unsqueeze(0).to(device)
            if x.shape[1] == context and y.shape[1] == context:
                _, loss = model(x, y)
                losses.append(float(loss.item()))
    val_loss = sum(losses) / len(losses) if losses else float("nan")
    report = {
        "project": "Independent AI Lab v0.2",
        "parameters": model.parameter_count(),
        "architecture": checkpoint["model_config"],
        "heldout_next_byte_loss": round(val_loss, 5),
        "heldout_byte_perplexity": round(float(torch.exp(torch.tensor(val_loss))), 4),
        "heldout_windows": len(losses),
        "training_provenance": checkpoint["training"],
        "limitations": [
            "This is a small research language model, not a ChatGPT-equivalent or frontier model.",
            "The model is trained on a small original educational corpus; factual breadth and reliability are limited.",
            "Byte-level perplexity measures next-byte prediction on this corpus and is not a general intelligence score.",
            "No image, audio, video, tool-use, or persistent-memory capability is trained in this release.",
        ],
    }
    out = ROOT / "evaluation" / "report_v0_2.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
