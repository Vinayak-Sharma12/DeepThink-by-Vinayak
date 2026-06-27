"""Split long token sequences into context-length windows."""

from __future__ import annotations


def chunk_token_ids(
    token_ids: list[int],
    ctx_len: int,
    *,
    add_bos: bool = False,
    add_eos: bool = False,
    bos_id: int | None = None,
    eos_id: int | None = None,
) -> list[list[int]]:
    """Split ``token_ids`` into windows of at most ``ctx_len`` tokens.

    When ``add_bos`` / ``add_eos`` are enabled, each chunk is wrapped with the
    corresponding special token ids. Chunks shorter than ``ctx_len`` are returned
    as-is (padding happens in the collator).
    """
    if ctx_len <= 0:
        msg = f"ctx_len must be positive, got {ctx_len}"
        raise ValueError(msg)
    if add_bos and bos_id is None:
        msg = "bos_id is required when add_bos=True"
        raise ValueError(msg)
    if add_eos and eos_id is None:
        msg = "eos_id is required when add_eos=True"
        raise ValueError(msg)
    if not token_ids:
        return []

    reserved = int(add_bos) + int(add_eos)
    body_len = max(ctx_len - reserved, 1)
    chunks: list[list[int]] = []
    for start in range(0, len(token_ids), body_len):
        body = token_ids[start : start + body_len]
        chunk: list[int] = []
        if add_bos:
            chunk.append(bos_id)  # type: ignore[arg-type]
        chunk.extend(body)
        if add_eos:
            chunk.append(eos_id)  # type: ignore[arg-type]
        chunks.append(chunk)
    return chunks
