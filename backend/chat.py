"""Chat with the locally trained model. No network calls are made."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ai_core.model import IndependentLanguageModel, ModelConfig  # noqa: E402
from tokenizer.byte_tokenizer import ByteTokenizer  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=str(ROOT / "checkpoints" / "independent_lm_v0_2.pt"))
    parser.add_argument("--max-new-tokens", type=int, default=180)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=35)
    args = parser.parse_args()
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise SystemExit(f"Checkpoint not found: {checkpoint_path}. Run training/train_v02.py first.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # weights_only=False is used because this is a checkpoint created by this project.
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ModelConfig(**checkpoint["model_config"])
    model = IndependentLanguageModel(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    tokenizer = ByteTokenizer()
    print("Independent AI Lab v0.2 (small research model; not a frontier assistant)")
    print("Type /quit to exit. This local prototype does not call external AI services.\n")

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break
        if question.lower() in {"/quit", "/exit"}:
            break
        if not question:
            continue
        prompt = f"USER: {question}\nASSISTANT:"
        encoded = tokenizer.encode(prompt)
        input_ids = torch.tensor([encoded], dtype=torch.long, device=device)
        output = model.generate(
            input_ids,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )
        generated = tokenizer.decode(output[0, len(encoded) :].tolist())
        # Keep the first assistant turn; the tiny training corpus may contain later turns.
        generated = generated.split("\nUSER:", 1)[0].strip()
        print(f"AI: {generated or '[No text generated; try a different prompt.]'}\n")


if __name__ == "__main__":
    main()
