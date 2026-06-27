"""Pack token streams into fixed context-length blocks."""

from __future__ import annotations


def pack_token_streams(
    sequences: list[list[int]],
    ctx_len: int,
    *,
    eos_id: int | None = None,
) -> list[list[int]]:
    """Concatenate token sequences and split into ``ctx_len`` blocks.

    When ``eos_id`` is provided, it is inserted between consecutive sequences
    before packing. The final block may be shorter than ``ctx_len``.
    """
    if ctx_len <= 0:
        msg = f"ctx_len must be positive, got {ctx_len}"
        raise ValueError(msg)

    stream: list[int] = []
    for index, sequence in enumerate(sequences):
        if not sequence:
            continue
        stream.extend(sequence)
        if eos_id is not None and index < len(sequences) - 1:
            stream.append(eos_id)

    if not stream:
        return []

    packed: list[list[int]] = []
    for start in range(0, len(stream), ctx_len):
        packed.append(stream[start : start + ctx_len])
    return packed
