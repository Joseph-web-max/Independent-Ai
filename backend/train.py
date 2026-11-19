"""Train the independent language-model baseline from randomly initialized weights."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ai_core.model import IndependentLanguageModel, ModelConfig  # noqa: E402
from tokenizer.byte_tokenizer import ByteTokenizer  # noqa: E402


def get_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but no CUDA GPU is available")
    return device


def get_batch(data: torch.Tensor, batch_size: int, context_length: int, device: torch.device):
    if len(data) <= context_length + 1:
        raise ValueError("Training split is too small for the selected context length")
    starts = torch.randint(0, len(data) - context_length - 1, (batch_size,))
    x = torch.stack([data[int(i) : int(i) + context_length] for i in starts])
    y = torch.stack([data[int(i) + 1 : int(i) + context_length + 1] for i in starts])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, train_data, val_data, batch_size, context_length, device, batches=4):
    model.eval()
    results = {}
    for name, data in (("train", train_data), ("validation", val_data)):
        losses = []
        for _ in range(batches):
            x, y = get_batch(data, batch_size, context_length, device)
            _, loss = model(x, y)
            losses.append(float(loss.item()))
        results[name] = sum(losses) / len(losses)
    model.train()
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20261118)
    parser.add_argument("--save-every", type=int, default=250)
    args = parser.parse_args()
    if args.steps < 1 or args.batch_size < 1:
        parser.error("--steps and --batch-size must be positive")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    device = get_device(args.device)

    corpus_path = ROOT / "data" / "seed_corpus.txt"
    corpus = corpus_path.read_text(encoding="utf-8")
    tokenizer = ByteTokenizer()
    token_ids = torch.tensor(tokenizer.encode(corpus), dtype=torch.long)
    split = int(0.9 * len(token_ids))
    train_data, val_data = token_ids[:split], token_ids[split:]

    config = ModelConfig(
        context_length=args.context_length,
        d_model=args.d_model,
        n_heads=args.heads,
        n_layers=args.layers,
        dropout=0.05,
    )
    model = IndependentLanguageModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.1)
    print(f"Device: {device}")
    print(f"Training bytes/tokens: {len(train_data):,}; validation: {len(val_data):,}")
    print(f"Parameters: {model.parameter_count():,}")
    print("Initialization: random; pretrained model weights: none")

    history = []
    start_time = time.time()
    baseline = estimate_loss(model, train_data, val_data, args.batch_size, args.context_length, device)
    print(f"Initial loss: train={baseline['train']:.4f}, validation={baseline['validation']:.4f}")

    model.train()
    for step in range(1, args.steps + 1):
        x, y = get_batch(train_data, args.batch_size, args.context_length, device)
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step == 1 or step % 50 == 0 or step == args.steps:
            metrics = estimate_loss(
                model, train_data, val_data, args.batch_size, args.context_length, device
            )
            record = {"step": step, **metrics, "elapsed_seconds": round(time.time() - start_time, 2)}
            history.append(record)
            print(
                f"step {step:4d}/{args.steps} | train {metrics['train']:.4f} "
                f"| validation {metrics['validation']:.4f} | {record['elapsed_seconds']:.1f}s"
            )
            if args.save_every > 0 and step % args.save_every == 0 and step < args.steps:
                checkpoint_dir = ROOT / "checkpoints"
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                snapshot = {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "model_config": model.config_dict(),
                    "step": step,
                    "training": {
                        "random_initialization": True,
                        "pretrained_weights_used": False,
                        "external_ai_apis_used": False,
                        "seed": args.seed,
                        "steps_completed": step,
                        "device": str(device),
                        "corpus_file": "data/seed_corpus.txt",
                        "corpus_bytes": len(token_ids),
                    },
                }
                torch.save(snapshot, checkpoint_dir / f"checkpoint_step_{step}.pt")

    final_metrics = estimate_loss(
        model, train_data, val_data, args.batch_size, args.context_length, device
    )
    elapsed = time.time() - start_time
    checkpoint_dir = ROOT / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "independent_lm_v0_1.pt"
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "model_config": model.config_dict(),
        "training": {
            "random_initialization": True,
            "pretrained_weights_used": False,
            "external_ai_apis_used": False,
            "seed": args.seed,
            "steps": args.steps,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "device": str(device),
            "initial_loss": baseline,
            "final_loss": final_metrics,
            "elapsed_seconds": round(elapsed, 2),
            "corpus_file": "data/seed_corpus.txt",
            "corpus_bytes": len(token_ids),
        },
    }
    torch.save(checkpoint, checkpoint_path)
    (ROOT / "training" / "training_history.json").write_text(
        json.dumps({"initial": baseline, "history": history, "final": final_metrics}, indent=2),
        encoding="utf-8",
    )
    tokenizer.save(str(ROOT / "tokenizer" / "tokenizer_config.json"))

    print("\nTraining complete.")
    print(f"Final loss: train={final_metrics['train']:.4f}, validation={final_metrics['validation']:.4f}")
    print(f"Elapsed: {elapsed:.1f}s")
    print(f"Checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
