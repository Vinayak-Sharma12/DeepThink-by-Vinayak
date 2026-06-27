"""Loss-mask utilities for pretrain and SFT batches."""

from __future__ import annotations


def build_shifted_targets(input_ids: list[int], pad_id: int) -> list[int]:
    """Build next-token targets aligned with ``input_ids``.

    For all valid positions ``t < len(input_ids) - 1``, ``targets[t] == input_ids[t + 1]``.
    The final position uses ``pad_id`` because there is no next token to predict.
    """
    if not input_ids:
        return []
    return list(input_ids[1:]) + [pad_id]


def build_loss_mask(prompt_len: int, total_len: int) -> list[int]:
    """Build a causal loss mask for prompt/completion training.

    Loss is computed at position ``t`` when predicting token ``t + 1``. Positions
    whose target still lies inside the prompt span (index ``< prompt_len``) are
    masked out. The final position is always masked because it has no target.

    For pretrain, pass ``prompt_len=0`` to count loss on every valid next-token
    position.
    """
    if prompt_len < 0:
        msg = f"prompt_len must be non-negative, got {prompt_len}"
        raise ValueError(msg)
    if total_len < 0:
        msg = f"total_len must be non-negative, got {total_len}"
        raise ValueError(msg)

    mask = [0] * total_len
    for position in range(total_len - 1):
        if position + 1 >= prompt_len:
            mask[position] = 1
    return mask


def build_pretrain_loss_mask(total_len: int) -> list[int]:
    """Return a loss mask with loss on every valid next-token position."""
    return build_loss_mask(prompt_len=0, total_len=total_len)


def apply_padding_to_loss_mask(
    loss_mask: list[int],
    pad_id: int,
    input_ids: list[int],
) -> list[int]:
    """Zero out loss-mask positions that correspond to padding tokens."""
    return [
        mask if token_id != pad_id else 0
        for mask, token_id in zip(loss_mask, input_ids, strict=True)
    ]
