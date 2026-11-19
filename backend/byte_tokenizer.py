"""A tiny, dependency-free UTF-8 byte tokenizer.

Every byte is a token (IDs 0..255). This deliberately avoids importing any
pretrained tokenizer vocabulary or model assets.
"""

from __future__ import annotations


class ByteTokenizer:
    vocab_size = 256

    def encode(self, text: str) -> list[int]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        return list(text.encode("utf-8", errors="replace"))

    def decode(self, token_ids: list[int]) -> str:
        safe_ids = [int(token_id) % 256 for token_id in token_ids]
        return bytes(safe_ids).decode("utf-8", errors="replace")

    def save(self, path: str) -> None:
        # The tokenizer has no learned vocabulary or external assets.
        with open(path, "w", encoding="utf-8") as f:
            f.write('{"type":"utf8_byte_level","vocab_size":256,"pretrained":false}\n')
