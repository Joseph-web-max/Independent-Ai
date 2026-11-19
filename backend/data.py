"""Deterministic document-level data split for Independent AI Lab v0.2."""
from __future__ import annotations

import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_document_splits(seed: int = 20261118, validation_fraction: float = 0.08) -> tuple[str, str]:
    """Keep the v0.1 seed corpus in training and hold out complete v0.2 documents."""
    if not 0.01 <= validation_fraction <= 0.3:
        raise ValueError("validation_fraction must be between 0.01 and 0.3")

    base_path = ROOT / "data" / "seed_corpus.txt"
    additions_path = ROOT / "data" / "v0_2_additions.txt"
    base = base_path.read_text(encoding="utf-8").strip()
    additions = additions_path.read_text(encoding="utf-8")
    docs = [part.strip() for part in additions.split("<|document|>") if "USER:" in part and "ASSISTANT:" in part]
    if len(docs) < 10:
        raise ValueError("v0.2 corpus needs at least 10 independent documents")

    rng = random.Random(seed)
    indices = list(range(len(docs)))
    rng.shuffle(indices)
    val_count = max(1, round(len(docs) * validation_fraction))
    val_indices = set(indices[:val_count])
    train_docs = [doc for i, doc in enumerate(docs) if i not in val_indices]
    val_docs = [doc for i, doc in enumerate(docs) if i in val_indices]

    train_text = base + "\n\n" + "\n\n".join(train_docs)
    val_text = "\n\n".join(val_docs)
    return train_text, val_text
