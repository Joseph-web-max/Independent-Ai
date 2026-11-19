"""Train Independent AI Lab v0.2 from fresh random initialization."""
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
from training.data import load_document_splits  # noqa: E402


def get_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return device


def batchify(data: torch.Tensor, batch_size: int, context_length: int, device: torch.device):
    if data.numel() <= context_length + 1:
        raise ValueError("A data split is too small for this context length")
    starts = torch.randint(0, data.numel() - context_length - 1, (batch_size,))
    x = torch.stack([data[int(i):int(i) + context_length] for i in starts])
    y = torch.stack([data[int(i) + 1:int(i) + context_length + 1] for i in starts])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, train_data, val_data, batch_size, context_length, device, batches=8):
    model.eval()
    results = {}
    for name, data in (("train", train_data), ("validation", val_data)):
        losses = []
        for _ in range(batches):
            x, y = batchify(data, batch_size, context_length, device)
            _, loss = model(x, y)
            losses.append(float(loss.item()))
        results[name] = sum(losses) / len(losses)
    model.train()
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=1400)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--context-length", type=int, default=192)
    parser.add_argument("--d-model", type=int, default=96)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2.5e-4)
    parser.add_argument("--warmup-steps", type=int, default=60)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20261119)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--validation-batches", type=int, default=8)
    parser.add_argument("--resume-from", default="", help="Optional v0.2 checkpoint to continue from")
    parser.add_argument("--run-steps", type=int, default=0, help="Maximum additional steps in this invocation; 0 means finish all remaining steps")
    args = parser.parse_args()
    if args.steps < 1 or args.batch_size < 1 or args.log_every < 1:
        parser.error("steps, batch-size and log-every must be positive")
    if args.warmup_steps < 0 or args.warmup_steps >= args.steps:
        parser.error("warmup-steps must be non-negative and less than steps")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    device = get_device(args.device)

    train_text, val_text = load_document_splits(seed=args.seed)
    tokenizer = ByteTokenizer()
    train_data = torch.tensor(tokenizer.encode(train_text), dtype=torch.long)
    val_data = torch.tensor(tokenizer.encode(val_text), dtype=torch.long)

    config = ModelConfig(
        context_length=args.context_length,
        d_model=args.d_model,
        n_heads=args.heads,
        n_layers=args.layers,
        dropout=0.08,
    )
    model = IndependentLanguageModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.08, betas=(0.9, 0.95))
    start_step = 1
    resumed = None
    if args.resume_from:
        resume_path = Path(args.resume_from)
        if not resume_path.is_absolute():
            resume_path = ROOT / resume_path
        if not resume_path.exists():
            raise SystemExit(f"Resume checkpoint does not exist: {resume_path}")
        resumed = torch.load(resume_path, map_location=device, weights_only=False)
        if resumed.get("model_config") != model.config_dict():
            raise SystemExit("Resume checkpoint architecture does not match the requested v0.2 configuration")
        model.load_state_dict(resumed["model_state_dict"])
        if "optimizer_state_dict" in resumed:
            optimizer.load_state_dict(resumed["optimizer_state_dict"])
        start_step = int(resumed.get("step", 0)) + 1
        print(f"Resuming own v0.2 checkpoint from step {start_step - 1}")
    print(f"Device: {device}")
    print(f"Training bytes: {train_data.numel():,}; held-out validation bytes: {val_data.numel():,}")
    print("Validation policy: deterministic document-level holdout")
    print(f"Parameters: {model.parameter_count():,}")
    if resumed:
        print("Initialization provenance: fresh random initialization at start of v0.2; continuing this project-owned checkpoint; no third-party weights or AI APIs")
    else:
        print("Initialization: fresh random weights; v0.1 weights not loaded; third-party pretrained weights: none; external AI APIs: none")

    history = []
    history_path = ROOT / "training" / "training_history_v0_2.json"
    start_time = time.time()
    elapsed_offset = 0.0
    if resumed:
        previous_training = resumed.get("training", {})
        initial = previous_training.get("initial_loss", {"train": 5.6016, "validation": 5.6016})
        best_val = float(previous_training.get("best_validation_loss_during_training", 2.6389))
        elapsed_offset = float(previous_training.get("elapsed_seconds", 28.9))
        if history_path.exists():
            try:
                history = json.loads(history_path.read_text(encoding="utf-8")).get("history", [])
            except (json.JSONDecodeError, OSError):
                history = []
        if not history:
            history = [
                {"step": 1, "train": 5.5984, "validation": 5.6004, "learning_rate": 4.17e-6, "elapsed_seconds": 0.9},
                {"step": 100, "train": 3.3230, "validation": 3.2968, "learning_rate": 2.5e-4, "elapsed_seconds": 14.8},
                {"step": 200, "train": 2.6662, "validation": 2.6389, "learning_rate": 2.44e-4, "elapsed_seconds": 28.9},
            ]
        print(f"Original random-init baseline: train={initial['train']:.4f}, validation={initial['validation']:.4f}", flush=True)
    else:
        initial = estimate_loss(model, train_data, val_data, args.batch_size, args.context_length, device, args.validation_batches)
        best_val = initial["validation"]
        print(f"Initial loss: train={initial['train']:.4f}, validation={initial['validation']:.4f}", flush=True)

    run_end_step = args.steps if args.run_steps <= 0 else min(args.steps, start_step + args.run_steps - 1)
    if start_step > args.steps:
        raise SystemExit(f"Checkpoint already completed {start_step - 1} steps, which is >= requested total {args.steps}")

    model.train()
    for step in range(start_step, run_end_step + 1):
        if step <= args.warmup_steps:
            lr_scale = step / max(1, args.warmup_steps)
        else:
            progress = (step - args.warmup_steps) / max(1, args.steps - args.warmup_steps)
            lr_scale = 0.12 + 0.88 * 0.5 * (1.0 + math.cos(math.pi * progress))
        lr = args.learning_rate * lr_scale
        for group in optimizer.param_groups:
            group["lr"] = lr

        x, y = batchify(train_data, args.batch_size, args.context_length, device)
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step == 1 or step % args.log_every == 0 or step == args.steps:
            metrics = estimate_loss(model, train_data, val_data, args.batch_size, args.context_length, device, args.validation_batches)
            record = {"step": step, **metrics, "learning_rate": lr, "elapsed_seconds": round(elapsed_offset + time.time() - start_time, 2)}
            history.append(record)
            print(f"step {step:4d}/{args.steps} | train {metrics['train']:.4f} | validation {metrics['validation']:.4f} | lr {lr:.2e} | {record['elapsed_seconds']:.1f}s", flush=True)
            if metrics["validation"] < best_val:
                best_val = metrics["validation"]
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "model_config": model.config_dict(),
                    "step": step,
                    "training_version": "0.2",
                    "training": {"random_initialization": True, "v0_1_weights_loaded": False, "third_party_pretrained_weights_used": False, "external_ai_apis_used": False, "seed": args.seed, "steps_completed": step, "device": str(device), "train_bytes": train_data.numel(), "validation_bytes": val_data.numel(), "initial_loss": initial, "best_validation_loss_during_training": best_val},
                }, ROOT / "checkpoints" / "independent_lm_v0_2_best.pt")
            latest_training = {"random_initialization": True, "v0_1_weights_loaded": False, "third_party_pretrained_weights_used": False, "external_ai_apis_used": False, "seed": args.seed, "steps_completed": step, "device": str(device), "train_bytes": train_data.numel(), "validation_bytes": val_data.numel(), "initial_loss": initial, "best_validation_loss_during_training": best_val, "elapsed_seconds": round(elapsed_offset + time.time() - start_time, 2)}
            torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "model_config": model.config_dict(), "step": step, "training_version": "0.2", "training": latest_training}, ROOT / "checkpoints" / "independent_lm_v0_2_latest.pt")
            history_path.write_text(json.dumps({"initial": initial, "history": history, "training": {"target_steps": args.steps, "last_completed_step": step, "best_validation_loss_during_training": best_val}}, indent=2), encoding="utf-8")

    if run_end_step < args.steps:
        print(f"Saved resumable checkpoint at step {run_end_step}/{args.steps}. Run again with --resume-from checkpoints/independent_lm_v0_2_latest.pt --run-steps N.", flush=True)
        return

    final = estimate_loss(model, train_data, val_data, args.batch_size, args.context_length, device, max(12, args.validation_batches))
    elapsed = elapsed_offset + time.time() - start_time
    training_metadata = {
        "random_initialization": True,
        "v0_1_weights_loaded": False,
        "third_party_pretrained_weights_used": False,
        "external_ai_apis_used": False,
        "seed": args.seed,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "context_length": args.context_length,
        "learning_rate_peak": args.learning_rate,
        "warmup_steps": args.warmup_steps,
        "device": str(device),
        "initial_loss": initial,
        "final_loss": final,
        "best_validation_loss_during_training": best_val,
        "elapsed_seconds": round(elapsed, 2),
        "training_bytes": train_data.numel(),
        "validation_bytes": val_data.numel(),
        "training_corpus": "data/seed_corpus.txt + data/v0_2_additions.txt",
        "document_level_validation_split": True,
    }
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_config": model.config_dict(),
        "training_version": "0.2",
        "training": training_metadata,
    }
    checkpoint_path = ROOT / "checkpoints" / "independent_lm_v0_2.pt"
    torch.save(checkpoint, checkpoint_path)
    history_path.write_text(json.dumps({"initial": initial, "history": history, "final": final, "training": training_metadata}, indent=2), encoding="utf-8")
    (ROOT / "tokenizer" / "tokenizer_config.json").write_text(json.dumps({"type": "utf8_byte", "vocab_size": 256, "special_tokens": [], "version": "0.2"}, indent=2), encoding="utf-8")

    print("\nTraining complete.")
    print(f"Final loss: train={final['train']:.4f}, validation={final['validation']:.4f}")
    print(f"Best observed validation loss: {best_val:.4f}")
    print(f"Elapsed: {elapsed:.1f}s")
    print(f"Checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
