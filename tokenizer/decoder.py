"""Token-id-to-text decoding for byte-level BPE."""

from __future__ import annotations

from tokenizer.vocab import Vocab


def decode_ids(
    token_ids: list[int],
    vocab: Vocab,
    *,
    skip_special: bool = True,
) -> str:
    """Decode token ids back to UTF-8 text."""
    parts: list[str | bytes] = []
    for token_id in token_ids:
        if vocab.is_special_id(token_id):
            if not skip_special:
                parts.append(vocab.id_to_special[token_id])
            continue
        parts.append(vocab.token_bytes(token_id))

    output: list[str] = []
    byte_buffer = bytearray()
    for part in parts:
        if isinstance(part, str):
            if byte_buffer:
                output.append(byte_buffer.decode("utf-8"))
                byte_buffer = bytearray()
            output.append(part)
        else:
            byte_buffer.extend(part)
    if byte_buffer:
        output.append(byte_buffer.decode("utf-8"))
    return "".join(output)
