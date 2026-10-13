"""Pure popularity-group and unweighted exposure metric primitives.

The caller supplies train-fit popularity counts, authoritative ranked item
IDs, and the complete eligible fit catalog.  This module does not load data,
sort scores, generate candidates, aggregate cohorts, or apply rank discounts.
"""

from collections.abc import Collection, Mapping, Sequence
from numbers import Integral
from operator import itemgetter

HEAD = "head"
MID_TAIL = "mid_tail"
LONG_TAIL = "long_tail"
_POPULARITY_GROUPS = frozenset({HEAD, MID_TAIL, LONG_TAIL})


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


def _validate_recommendations(
    recommendations: Sequence[Integral],
) -> tuple[Integral, ...]:
    if not isinstance(recommendations, Sequence):
        raise TypeError("each recommendation list must be an ordered sequence")

    validated: list[Integral] = []
    seen: set[Integral] = set()
    for item_id in recommendations:
        _validate_item_id(item_id, input_name="recommendations")
        if item_id in seen:
            raise ValueError("recommendations must not contain duplicate item IDs")
        seen.add(item_id)
        validated.append(item_id)
    return tuple(validated)


def _validate_recommendation_lists(
    recommendation_lists: Collection[Sequence[Integral]],
) -> tuple[tuple[Integral, ...], ...]:
    if not isinstance(recommendation_lists, Collection):
        raise TypeError("recommendation_lists must be a collection")
    return tuple(
        _validate_recommendations(recommendations)
        for recommendations in recommendation_lists
    )


def _validate_unique_catalog(
    catalog_items: Collection[Integral],
) -> tuple[Integral, ...]:
    if not isinstance(catalog_items, Collection):
        raise TypeError("catalog_items must be a collection")

    validated: list[Integral] = []
    seen: set[Integral] = set()
    for item_id in catalog_items:
        _validate_item_id(item_id, input_name="catalog_items")
        if item_id in seen:
            raise ValueError("catalog_items must not contain duplicate item IDs")
        seen.add(item_id)
        validated.append(item_id)

    if not validated:
        raise ValueError("catalog_items must not be empty")
    return tuple(validated)


def _validate_popularity_groups(
    popularity_groups: Mapping[Integral, str],
) -> dict[Integral, str]:
    if not isinstance(popularity_groups, Mapping):
        raise TypeError("popularity_groups must be a mapping")
    if not popularity_groups:
        raise ValueError("popularity_groups must not be empty")

    validated: dict[Integral, str] = {}
    for item_id, group in popularity_groups.items():
        _validate_item_id(item_id, input_name="popularity_groups")
        if not isinstance(group, str):
            raise TypeError("popularity group labels must be strings")
        if group not in _POPULARITY_GROUPS:
            raise ValueError(
                "popularity group labels must be head, mid_tail, or long_tail"
            )
        validated[item_id] = group
    return validated


def assign_popularity_groups(
    item_positive_counts: Mapping[Integral, Integral],
) -> dict[Integral, str]:
    """Assign every fit-catalog item to a deterministic popularity group.

    Items are ordered by ``positive_count`` descending and then item ID
    ascending.  For catalog size ``n``, the frozen item-count boundaries are
    ``(n * 20) // 100`` and ``(n * 50) // 100``.  Exact slicing is retained
    for small catalogs, even when one of the first slices is empty.
    """

    if not isinstance(item_positive_counts, Mapping):
        raise TypeError("item_positive_counts must be a mapping")
    if not item_positive_counts:
        raise ValueError("item_positive_counts must not be empty")

    validated: list[tuple[Integral, Integral]] = []
    for item_id, count in item_positive_counts.items():
        _validate_item_id(item_id, input_name="item_positive_counts")
        if isinstance(count, bool) or not isinstance(count, Integral):
            raise TypeError(
                "item_positive_counts values must be non-negative integer scalars"
            )
        if count < 0:
            raise ValueError("item_positive_counts values must be non-negative")
        validated.append((item_id, count))

    ordered_items_and_counts = sorted(validated)
    ordered_items_and_counts.sort(key=itemgetter(1), reverse=True)
    ordered_items = [item_id for item_id, _ in ordered_items_and_counts]
    catalog_size = len(ordered_items)
    head_end = (catalog_size * 20) // 100
    mid_end = (catalog_size * 50) // 100

    groups: dict[Integral, str] = {}
    groups.update(dict.fromkeys(ordered_items[:head_end], HEAD))
    groups.update(dict.fromkeys(ordered_items[head_end:mid_end], MID_TAIL))
    groups.update(dict.fromkeys(ordered_items[mid_end:], LONG_TAIL))
    return groups


def long_tail_exposure_at_k(
    recommendation_lists: Collection[Sequence[Integral]],
    popularity_groups: Mapping[Integral, str],
    k: int,
) -> float:
    """Return long-tail top-``k`` slots divided by actual displayed slots.

    Each returned top-``k`` position has equal weight.  Short lists contribute
    only their available positions, and no displayed positions return
    ``0.0``.  Recommendation membership and duplicates are validated across
    every complete list, including positions below ``k``.
    """

    validated_k = _validate_k(k)
    validated_lists = _validate_recommendation_lists(recommendation_lists)
    groups = _validate_popularity_groups(popularity_groups)

    for recommendations in validated_lists:
        for item_id in recommendations:
            if item_id not in groups:
                raise ValueError("recommendation item is absent from popularity_groups")

    displayed_slots = sum(
        min(len(recommendations), validated_k) for recommendations in validated_lists
    )
    if displayed_slots == 0:
        return 0.0

    long_tail_slots = sum(
        groups[item_id] == LONG_TAIL
        for recommendations in validated_lists
        for item_id in recommendations[:validated_k]
    )
    return long_tail_slots / displayed_slots


def exposure_gini_at_k(
    recommendation_lists: Collection[Sequence[Integral]],
    catalog_items: Collection[Integral],
    k: int,
) -> float:
    """Return standard Gini over complete fit-catalog top-``k`` exposures.

    The exposure vector contains every eligible fit-catalog item, including
    items with zero exposure.  Every displayed top-``k`` position has equal
    weight.  No displayed positions return ``0.0``; no finite-maximum
    normalization is applied.
    """

    validated_k = _validate_k(k)
    validated_lists = _validate_recommendation_lists(recommendation_lists)
    catalog = _validate_unique_catalog(catalog_items)
    exposure_counts = dict.fromkeys(catalog, 0)

    for recommendations in validated_lists:
        for item_id in recommendations:
            if item_id not in exposure_counts:
                raise ValueError("recommendation item is absent from catalog_items")
        for item_id in recommendations[:validated_k]:
            exposure_counts[item_id] += 1

    exposure_vector = list(exposure_counts.values())
    total_exposure = sum(exposure_vector)
    if total_exposure == 0:
        return 0.0

    sorted_exposures = sorted(exposure_vector)
    catalog_size = len(sorted_exposures)
    numerator = sum(
        (2 * rank - catalog_size - 1) * exposure
        for rank, exposure in enumerate(sorted_exposures, start=1)
    )
    return numerator / (catalog_size * total_exposure)
