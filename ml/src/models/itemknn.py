"""Pure deterministic sparse ItemKNN training and recommendation helpers.

The similarity orientation is ``S[source_item, candidate_item]``.  Top-neighbor
pruning is therefore allowed to make the stored graph directed.  This module
fits exactly one caller-supplied neighbor count; it does not tune ItemKNN.
The value 100 used by MODEL-002's smoke config is not selected or preferred.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Sequence
from dataclasses import dataclass
from numbers import Integral

import numpy as np
from scipy.sparse import csr_matrix, isspmatrix_csr


def _validate_external_id(value: Integral, *, input_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{input_name} IDs must be non-negative integer scalars")
    if value < 0:
        raise ValueError(f"{input_name} IDs must be non-negative")
    return int(value)


def _validate_id_sequence(
    values: Sequence[Integral],
    *,
    input_name: str,
    allow_empty: bool = False,
) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{input_name} must be an ordered sequence")
    validated = tuple(
        _validate_external_id(value, input_name=input_name) for value in values
    )
    if not validated and not allow_empty:
        raise ValueError(f"{input_name} must not be empty")
    if len(validated) != len(set(validated)):
        raise ValueError(f"{input_name} must not contain duplicate IDs")
    return validated


def _validate_positive_integer(value: int, *, input_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{input_name} must be an integer scalar")
    if value <= 0:
        raise ValueError(f"{input_name} must be greater than zero")
    return int(value)


@dataclass(frozen=True)
class ItemKNNModel:
    """Sparse directed neighbor graph aligned to canonical external item IDs."""

    item_ids: tuple[int, ...]
    similarities: csr_matrix
    n_neighbors: int


def build_binary_user_item_matrix(
    interactions: Iterable[tuple[Integral, Integral]],
    user_ids: Sequence[Integral],
    item_ids: Sequence[Integral],
) -> csr_matrix:
    """Build a binary CSR matrix aligned to the supplied fit mappings.

    Each interaction is an external ``(user_id, item_id)`` pair.  Duplicate
    pairs and IDs absent from the supplied fit-only mappings fail fast.
    """

    users = _validate_id_sequence(user_ids, input_name="user_ids")
    items = _validate_id_sequence(item_ids, input_name="item_ids")
    if isinstance(interactions, (str, bytes)) or not isinstance(interactions, Iterable):
        raise TypeError("interactions must be an iterable of ID pairs")

    user_indices = {user_id: index for index, user_id in enumerate(users)}
    item_indices = {item_id: index for index, item_id in enumerate(items)}
    rows: list[int] = []
    columns: list[int] = []
    observed: set[tuple[int, int]] = set()

    for interaction in interactions:
        if not isinstance(interaction, (tuple, list)) or len(interaction) != 2:
            raise TypeError(
                "interactions must contain two-element (user_id, item_id) pairs"
            )
        user_id = _validate_external_id(interaction[0], input_name="interaction user")
        item_id = _validate_external_id(interaction[1], input_name="interaction item")
        pair = (user_id, item_id)
        if pair in observed:
            raise ValueError("interactions must not contain duplicate pairs")
        observed.add(pair)
        if user_id not in user_indices:
            raise ValueError(f"interaction user ID is absent from mapping: {user_id}")
        if item_id not in item_indices:
            raise ValueError(f"interaction item ID is absent from mapping: {item_id}")
        rows.append(user_indices[user_id])
        columns.append(item_indices[item_id])

    data = np.ones(len(rows), dtype=np.float64)
    matrix = csr_matrix(
        (data, (rows, columns)),
        shape=(len(users), len(items)),
        dtype=np.float64,
    )
    matrix.sort_indices()
    return matrix


def _validate_binary_matrix(
    user_item_matrix: csr_matrix,
    *,
    item_count: int,
) -> None:
    if not isspmatrix_csr(user_item_matrix):
        raise TypeError("user_item_matrix must be a SciPy CSR matrix")
    if user_item_matrix.ndim != 2:
        raise ValueError("user_item_matrix must be two-dimensional")
    if user_item_matrix.shape[0] <= 0 or user_item_matrix.shape[1] <= 0:
        raise ValueError("user_item_matrix dimensions must be positive")
    if user_item_matrix.shape[1] != item_count:
        raise ValueError("user_item_matrix column count must match item_ids length")
    if user_item_matrix.nnz == 0:
        raise ValueError("user_item_matrix must contain at least one interaction")
    if not user_item_matrix.has_canonical_format:
        raise ValueError(
            "user_item_matrix must use canonical CSR without duplicate entries"
        )
    if not np.isfinite(user_item_matrix.data).all():
        raise ValueError("user_item_matrix values must be finite")
    if not np.equal(user_item_matrix.data, 1.0).all():
        raise ValueError("user_item_matrix must contain only binary value 1.0")


def _prune_neighbors(
    similarities: csr_matrix,
    item_ids: np.ndarray,
    n_neighbors: int,
) -> csr_matrix:
    row_indices: list[int] = []
    column_indices: list[int] = []
    values: list[float] = []

    for source_index in range(similarities.shape[0]):
        start = similarities.indptr[source_index]
        end = similarities.indptr[source_index + 1]
        candidate_indices = similarities.indices[start:end]
        candidate_scores = similarities.data[start:end]
        if not len(candidate_indices):
            continue
        order = np.lexsort((item_ids[candidate_indices], -candidate_scores))
        selected = order[:n_neighbors]
        row_indices.extend([source_index] * len(selected))
        column_indices.extend(candidate_indices[selected].tolist())
        values.extend(candidate_scores[selected].tolist())

    graph = csr_matrix(
        (values, (row_indices, column_indices)),
        shape=similarities.shape,
        dtype=np.float64,
    )
    graph.eliminate_zeros()
    graph.sort_indices()
    return graph


def fit_itemknn(
    user_item_matrix: csr_matrix,
    item_ids: Sequence[Integral],
    n_neighbors: int,
) -> ItemKNNModel:
    """Fit cosine neighbors without materializing a dense item-item matrix."""

    supplied_item_ids = _validate_id_sequence(item_ids, input_name="item_ids")
    validated_neighbors = _validate_positive_integer(
        n_neighbors, input_name="n_neighbors"
    )
    _validate_binary_matrix(
        user_item_matrix,
        item_count=len(supplied_item_ids),
    )

    canonical_order = np.argsort(
        np.asarray(supplied_item_ids, dtype=np.int64),
        kind="stable",
    )
    canonical_ids = tuple(supplied_item_ids[index] for index in canonical_order)
    matrix = user_item_matrix[:, canonical_order].astype(np.float64, copy=True)
    matrix.sort_indices()

    item_counts = matrix.getnnz(axis=0).astype(np.float64, copy=False)
    item_norms = np.sqrt(item_counts)
    cooccurrence = (matrix.T @ matrix).tocsr()
    cooccurrence.setdiag(0.0)
    cooccurrence.eliminate_zeros()
    cooccurrence.sort_indices()

    repeated_source_norms = np.repeat(
        item_norms,
        np.diff(cooccurrence.indptr),
    )
    denominators = repeated_source_norms * item_norms[cooccurrence.indices]
    cooccurrence.data = cooccurrence.data / denominators
    if not np.isfinite(cooccurrence.data).all():
        raise ValueError("cosine normalization produced a non-finite value")
    cooccurrence.data = np.clip(cooccurrence.data, 0.0, 1.0)
    cooccurrence.eliminate_zeros()

    graph = _prune_neighbors(
        cooccurrence,
        np.asarray(canonical_ids, dtype=np.int64),
        validated_neighbors,
    )
    return ItemKNNModel(
        item_ids=canonical_ids,
        similarities=graph,
        n_neighbors=validated_neighbors,
    )


def _validate_model(model: ItemKNNModel) -> dict[int, int]:
    if not isinstance(model, ItemKNNModel):
        raise TypeError("model must be an ItemKNNModel")
    item_ids = _validate_id_sequence(model.item_ids, input_name="model item_ids")
    _validate_positive_integer(model.n_neighbors, input_name="model n_neighbors")
    similarities = model.similarities
    if not isspmatrix_csr(similarities):
        raise TypeError("model similarities must be a SciPy CSR matrix")
    expected_shape = (len(item_ids), len(item_ids))
    if similarities.shape != expected_shape:
        raise ValueError("model similarity shape must match model item_ids")
    if not similarities.has_canonical_format:
        raise ValueError("model similarities must use canonical CSR")
    if not np.isfinite(similarities.data).all():
        raise ValueError("model similarities must be finite")
    if np.any(similarities.data <= 0.0) or np.any(similarities.data > 1.0):
        raise ValueError("model similarities must be in the interval (0, 1]")
    if similarities.diagonal().any():
        raise ValueError("model similarities must not contain self-neighbors")
    row_counts = np.diff(similarities.indptr)
    if np.any(row_counts > model.n_neighbors):
        raise ValueError("model similarity row exceeds n_neighbors")
    return {item_id: index for index, item_id in enumerate(item_ids)}


def _validate_profile(
    positive_profile_items: Collection[Integral],
    item_indices: dict[int, int],
) -> tuple[int, ...]:
    if isinstance(positive_profile_items, (str, bytes)) or not isinstance(
        positive_profile_items, Collection
    ):
        raise TypeError("positive_profile_items must be a collection")
    profile: list[int] = []
    observed: set[int] = set()
    for value in positive_profile_items:
        item_id = _validate_external_id(value, input_name="positive_profile_items")
        if item_id in observed:
            raise ValueError("positive_profile_items must not contain duplicate IDs")
        if item_id not in item_indices:
            raise ValueError(
                f"positive profile item is absent from fit catalog: {item_id}"
            )
        observed.add(item_id)
        profile.append(item_indices[item_id])
    return tuple(profile)


def score_itemknn_candidates(
    model: ItemKNNModel,
    positive_profile_items: Collection[Integral],
) -> np.ndarray:
    """Sum outgoing neighbor similarities from the binary positive profile."""

    item_indices = _validate_model(model)
    profile_indices = _validate_profile(
        positive_profile_items,
        item_indices,
    )
    scores = np.zeros(len(model.item_ids), dtype=np.float64)
    for source_index in profile_indices:
        start = model.similarities.indptr[source_index]
        end = model.similarities.indptr[source_index + 1]
        candidates = model.similarities.indices[start:end]
        scores[candidates] += model.similarities.data[start:end]
    return scores


def recommend_itemknn(
    model: ItemKNNModel,
    positive_profile_items: Collection[Integral],
    previously_rated_items: Collection[Integral],
    k: int,
) -> tuple[tuple[int, float], ...]:
    """Return positive-score unseen candidates by score then movie ID."""

    validated_k = _validate_positive_integer(k, input_name="k")
    item_indices = _validate_model(model)
    _validate_profile(positive_profile_items, item_indices)
    if isinstance(previously_rated_items, (str, bytes)) or not isinstance(
        previously_rated_items, Collection
    ):
        raise TypeError("previously_rated_items must be a collection")

    rated: set[int] = set()
    for value in previously_rated_items:
        item_id = _validate_external_id(value, input_name="previously_rated_items")
        if item_id in rated:
            raise ValueError("previously_rated_items must not contain duplicate IDs")
        rated.add(item_id)

    if not positive_profile_items:
        return ()
    scores = score_itemknn_candidates(model, positive_profile_items)
    candidates = [
        (item_id, float(scores[index]))
        for index, item_id in enumerate(model.item_ids)
        if item_id not in rated and np.isfinite(scores[index]) and scores[index] > 0.0
    ]
    candidates.sort(key=lambda pair: (-pair[1], pair[0]))
    return tuple(candidates[:validated_k])
