"""Pure beyond-accuracy metrics over authoritative ranked item IDs.

The caller prepares the fit catalog, train-fit popularity counts, and genre
collections.  This module does not load data, generate candidates, sort
scores, aggregate user-level metrics, or parse MovieLens metadata.
"""

import math
from collections.abc import Collection, Mapping, Sequence
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


def _validate_recommendations(
    recommendations: Sequence[Integral],
) -> tuple[Integral, ...]:
    if not isinstance(recommendations, Sequence):
        raise TypeError("recommendations must be an ordered sequence")

    validated: list[Integral] = []
    seen: set[Integral] = set()
    for item_id in recommendations:
        _validate_item_id(item_id, input_name="recommendations")
        if item_id in seen:
            raise ValueError("recommendations must not contain duplicate item IDs")
        seen.add(item_id)
        validated.append(item_id)
    return tuple(validated)


def _validate_unique_item_collection(
    items: Collection[Integral],
    *,
    input_name: str,
) -> tuple[Integral, ...]:
    if not isinstance(items, Collection):
        raise TypeError(f"{input_name} must be a collection")

    validated: list[Integral] = []
    seen: set[Integral] = set()
    for item_id in items:
        _validate_item_id(item_id, input_name=input_name)
        if item_id in seen:
            raise ValueError(f"{input_name} must not contain duplicate item IDs")
        seen.add(item_id)
        validated.append(item_id)
    return tuple(validated)


def catalog_coverage_at_k(
    recommendation_lists: Collection[Sequence[Integral]],
    catalog_items: Collection[Integral],
    k: int,
) -> float:
    """Return global ``|union_u Rec_u[:k]| / |fit catalog|``.

    Empty recommendation collections and empty individual lists are valid and
    yield zero coverage when nothing is recommended.  The fit catalog must be
    non-empty and contain unique, non-negative integer item IDs.  Every item
    contributing within top ``k`` must belong to that catalog.  Duplicate IDs
    are rejected within each complete user list, including below ``k``.
    """

    validated_k = _validate_k(k)
    if not isinstance(recommendation_lists, Collection):
        raise TypeError("recommendation_lists must be a collection")

    catalog = _validate_unique_item_collection(
        catalog_items,
        input_name="catalog_items",
    )
    if not catalog:
        raise ValueError("catalog_items must not be empty")
    catalog_set = frozenset(catalog)

    recommended_top_k: set[Integral] = set()
    for recommendations in recommendation_lists:
        ranked = _validate_recommendations(recommendations)
        for item_id in ranked[:validated_k]:
            if item_id not in catalog_set:
                raise ValueError(
                    "top-k recommendation item is absent from catalog_items"
                )
            recommended_top_k.add(item_id)

    return len(recommended_top_k) / len(catalog)


def _validate_positive_counts(
    item_positive_counts: Mapping[Integral, Integral],
) -> dict[Integral, Integral]:
    if not isinstance(item_positive_counts, Mapping):
        raise TypeError("item_positive_counts must be a mapping")
    if not item_positive_counts:
        raise ValueError("item_positive_counts must not be empty")

    validated: dict[Integral, Integral] = {}
    for item_id, count in item_positive_counts.items():
        _validate_item_id(item_id, input_name="item_positive_counts")
        if isinstance(count, bool) or not isinstance(count, Integral):
            raise TypeError(
                "item_positive_counts values must be non-negative integer scalars"
            )
        if count < 0:
            raise ValueError("item_positive_counts values must be non-negative")
        validated[item_id] = count
    return validated


def novelty_at_k(
    recommendations: Sequence[Integral],
    item_positive_counts: Mapping[Integral, Integral],
    k: int,
) -> float:
    """Return mean smoothed self-information over returned top-``k`` items.

    For every supplied fit-catalog item, ``p_i = (n_i + 1) /
    sum_j(n_j + 1)`` and self-information is ``-log2(p_i)``.  A short list is
    divided by its returned-item count, not by ``k``; an empty list returns
    ``0.0``.  Counts and membership are validated across the complete inputs,
    including recommendation positions below ``k``.
    """

    validated_k = _validate_k(k)
    ranked = _validate_recommendations(recommendations)
    counts = _validate_positive_counts(item_positive_counts)

    for item_id in ranked:
        if item_id not in counts:
            raise ValueError("recommendation item is absent from item_positive_counts")

    top_k = ranked[:validated_k]
    if not top_k:
        return 0.0

    smoothed_mass = sum(int(count) + 1 for count in counts.values())
    log_mass = math.log2(smoothed_mass)
    self_information = (
        log_mass - math.log2(int(counts[item_id]) + 1) for item_id in top_k
    )
    return math.fsum(self_information) / len(top_k)


def _validate_item_genres(
    item_genres: Mapping[Integral, Collection[str]],
) -> dict[Integral, frozenset[str]]:
    if not isinstance(item_genres, Mapping):
        raise TypeError("item_genres must be a mapping")

    validated: dict[Integral, frozenset[str]] = {}
    for item_id, genres in item_genres.items():
        _validate_item_id(item_id, input_name="item_genres")
        if isinstance(genres, (str, bytes)) or not isinstance(
            genres,
            Collection,
        ):
            raise TypeError("item_genres values must be genre-label collections")

        seen: set[str] = set()
        for label in genres:
            if not isinstance(label, str):
                raise TypeError("genre labels must be strings")
            if label == "":
                raise ValueError("genre labels must not be empty strings")
            if label in seen:
                raise ValueError("genre collections must not contain duplicate labels")
            seen.add(label)
        validated[item_id] = frozenset(seen)
    return validated


def _jaccard_distance(
    left: frozenset[str],
    right: frozenset[str],
) -> float:
    union = left | right
    if not union:
        return 0.0
    return 1.0 - len(left & right) / len(union)


def intra_list_diversity_at_k(
    recommendations: Sequence[Integral],
    item_genres: Mapping[Integral, Collection[str]],
    k: int,
) -> float:
    """Return unweighted mean Jaccard distance over unordered top-``k`` pairs.

    Fewer than two returned items yield ``0.0``.  Two empty genre sets have
    distance ``0.0``; exactly one empty set has distance ``1.0``.  Genre
    metadata and recommendation membership are validated across the complete
    supplied inputs, including positions below ``k``.
    """

    validated_k = _validate_k(k)
    ranked = _validate_recommendations(recommendations)
    genres_by_item = _validate_item_genres(item_genres)

    for item_id in ranked:
        if item_id not in genres_by_item:
            raise ValueError("recommendation item is absent from item_genres")

    top_k = ranked[:validated_k]
    if len(top_k) < 2:
        return 0.0

    distances = (
        _jaccard_distance(
            genres_by_item[top_k[left_index]],
            genres_by_item[top_k[right_index]],
        )
        for left_index in range(len(top_k) - 1)
        for right_index in range(left_index + 1, len(top_k))
    )
    pair_count = len(top_k) * (len(top_k) - 1) // 2
    return math.fsum(distances) / pair_count
