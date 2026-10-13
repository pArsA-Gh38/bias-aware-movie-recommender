"""Exact per-user top-K ranking metrics for binary relevance.

The caller supplies an authoritative ranked sequence of item IDs and a binary
relevant-item collection.  This module does not sort, generate, or filter
recommendations.
"""

import math
from collections.abc import Collection, Sequence
from numbers import Integral


def _validate_k(k: int) -> int:
    if isinstance(k, bool) or not isinstance(k, Integral):
        raise TypeError("k must be an integer scalar, not a boolean")
    if k <= 0:
        raise ValueError("k must be greater than zero")
    return int(k)


def _validate_item_id(item_id: Integral, *, input_name: str) -> None:
    if isinstance(item_id, bool) or not isinstance(item_id, Integral):
        raise TypeError(f"{input_name} item IDs must be non-negative integer scalars")
    if item_id < 0:
        raise ValueError(f"{input_name} item IDs must be non-negative")


def _validate_unique_items(
    items: Collection[Integral],
    *,
    input_name: str,
) -> tuple[Integral, ...]:
    validated: list[Integral] = []
    seen: set[Integral] = set()

    for item_id in items:
        _validate_item_id(item_id, input_name=input_name)
        if item_id in seen:
            raise ValueError(f"{input_name} must not contain duplicate item IDs")
        seen.add(item_id)
        validated.append(item_id)

    return tuple(validated)


def _validate_inputs(
    recommendations: Sequence[Integral],
    relevant_items: Collection[Integral],
    k: int,
) -> tuple[tuple[Integral, ...], frozenset[Integral], int]:
    validated_k = _validate_k(k)

    if not isinstance(recommendations, Sequence):
        raise TypeError("recommendations must be an ordered sequence")
    if not isinstance(relevant_items, Collection):
        raise TypeError("relevant_items must be a collection")

    ranked = _validate_unique_items(
        recommendations,
        input_name="recommendations",
    )
    relevant = frozenset(
        _validate_unique_items(
            relevant_items,
            input_name="relevant_items",
        )
    )
    return ranked, relevant, validated_k


def _hit_count(
    recommendations: tuple[Integral, ...],
    relevant_items: frozenset[Integral],
    k: int,
) -> int:
    return sum(item_id in relevant_items for item_id in recommendations[:k])


def precision_at_k(
    recommendations: Sequence[Integral],
    relevant_items: Collection[Integral],
    k: int,
) -> float:
    """Return binary ``|Rec[:k] ∩ Rel| / k``.

    Sequence position defines rank 1 onward.  The denominator remains ``k``
    for a short list.  An empty relevant set returns ``0.0``.  Duplicate IDs
    in either input are rejected, including recommendations below ``k``.
    Invalid ``k`` types/IDs raise ``TypeError``; non-positive ``k``, negative
    IDs, and duplicates raise ``ValueError``.
    """

    ranked, relevant, validated_k = _validate_inputs(
        recommendations,
        relevant_items,
        k,
    )
    return _hit_count(ranked, relevant, validated_k) / validated_k


def recall_at_k(
    recommendations: Sequence[Integral],
    relevant_items: Collection[Integral],
    k: int,
) -> float:
    """Return binary ``|Rec[:k] ∩ Rel| / |Rel|``.

    Sequence position defines rank 1 onward.  An empty relevant set returns
    ``0.0`` instead of dividing by zero.  Duplicate IDs in either input are
    rejected, including recommendations below ``k``.  Invalid ``k`` types/IDs
    raise ``TypeError``; non-positive ``k``, negative IDs, and duplicates
    raise ``ValueError``.
    """

    ranked, relevant, validated_k = _validate_inputs(
        recommendations,
        relevant_items,
        k,
    )
    if not relevant:
        return 0.0
    return _hit_count(ranked, relevant, validated_k) / len(relevant)


def hit_at_k(
    recommendations: Sequence[Integral],
    relevant_items: Collection[Integral],
    k: int,
) -> float:
    """Return the per-user binary hit indicator at ``k``.

    The result is ``1.0`` when any rank 1 through ``k`` is relevant and
    ``0.0`` otherwise, including an empty relevant set.  Duplicate IDs in
    either input are rejected, including recommendations below ``k``.
    Invalid ``k`` types/IDs raise ``TypeError``; non-positive ``k``, negative
    IDs, and duplicates raise ``ValueError``.
    """

    ranked, relevant, validated_k = _validate_inputs(
        recommendations,
        relevant_items,
        k,
    )
    return float(_hit_count(ranked, relevant, validated_k) > 0)


def ndcg_at_k(
    recommendations: Sequence[Integral],
    relevant_items: Collection[Integral],
    k: int,
) -> float:
    """Return binary NDCG at ``k`` with ranks starting at 1.

    ``DCG = Σ rel_r/log2(r+1)`` over supplied ranks through ``k``.
    ``IDCG`` is independent of the actual list and uses
    ``min(k, |Rel|)`` ideal hits.  An empty relevant set has zero IDCG and
    returns ``0.0``.  Duplicate IDs in either input are rejected, including
    recommendations below ``k``.  Invalid ``k`` types/IDs raise ``TypeError``;
    non-positive ``k``, negative IDs, and duplicates raise ``ValueError``.
    """

    ranked, relevant, validated_k = _validate_inputs(
        recommendations,
        relevant_items,
        k,
    )
    ideal_hits = min(validated_k, len(relevant))
    if ideal_hits == 0:
        return 0.0

    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, item_id in enumerate(
            ranked[:validated_k],
            start=1,
        )
        if item_id in relevant
    )
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg
