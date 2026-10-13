"""Pure deterministic global-popularity ranking and recommendation helpers."""

from collections.abc import Collection, Mapping, Sequence
from numbers import Integral


def _validate_item_id(item_id: Integral, *, input_name: str) -> None:
    if isinstance(item_id, bool) or not isinstance(item_id, Integral):
        raise TypeError(f"{input_name} item IDs must be non-negative integer scalars")
    if item_id < 0:
        raise ValueError(f"{input_name} item IDs must be non-negative")


def _validate_k(k: int) -> int:
    if isinstance(k, bool) or not isinstance(k, Integral):
        raise TypeError("k must be an integer scalar, not a boolean")
    if k <= 0:
        raise ValueError("k must be greater than zero")
    return int(k)


def build_popularity_ranking(
    item_positive_counts: Mapping[Integral, Integral],
) -> tuple[Integral, ...]:
    """Rank the complete fit catalog by count descending and item ID ascending.

    Counts must be strictly positive because this baseline's fit catalog is
    defined by items present in positive fit interactions.
    """

    if not isinstance(item_positive_counts, Mapping):
        raise TypeError("item_positive_counts must be a mapping")
    if not item_positive_counts:
        raise ValueError("item_positive_counts must not be empty")

    validated: list[tuple[Integral, int]] = []
    for item_id, count in item_positive_counts.items():
        _validate_item_id(item_id, input_name="item_positive_counts")
        if isinstance(count, bool) or not isinstance(count, Integral):
            raise TypeError(
                "item_positive_counts values must be positive integer scalars"
            )
        if count <= 0:
            raise ValueError("item_positive_counts values must be greater than zero")
        validated.append((item_id, int(count)))

    validated.sort(key=lambda pair: (-pair[1], int(pair[0])))
    return tuple(item_id for item_id, _ in validated)


def recommend_popular_items(
    popularity_ranking: Sequence[Integral],
    previously_rated_items: Collection[Integral],
    k: int,
) -> tuple[Integral, ...]:
    """Return the first ``k`` unseen items while preserving global order.

    Seen-item filtering is deliberately applied before top-K truncation.
    Valid historical item IDs outside the fit catalog are harmless.
    """

    validated_k = _validate_k(k)
    if not isinstance(popularity_ranking, Sequence):
        raise TypeError("popularity_ranking must be an ordered sequence")
    if isinstance(previously_rated_items, (str, bytes)) or not isinstance(
        previously_rated_items,
        Collection,
    ):
        raise TypeError("previously_rated_items must be a collection")

    ranking: list[Integral] = []
    ranked_seen: set[Integral] = set()
    for item_id in popularity_ranking:
        _validate_item_id(item_id, input_name="popularity_ranking")
        if item_id in ranked_seen:
            raise ValueError("popularity_ranking must not contain duplicate item IDs")
        ranked_seen.add(item_id)
        ranking.append(item_id)

    rated: set[Integral] = set()
    for item_id in previously_rated_items:
        _validate_item_id(item_id, input_name="previously_rated_items")
        if item_id in rated:
            raise ValueError(
                "previously_rated_items must not contain duplicate item IDs"
            )
        rated.add(item_id)

    recommendations: list[Integral] = []
    for item_id in ranking:
        if item_id in rated:
            continue
        recommendations.append(item_id)
        if len(recommendations) == validated_k:
            break
    return tuple(recommendations)
