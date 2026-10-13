from __future__ import annotations

import hashlib
import math

import numpy as np
import pytest

from src.evaluation.bootstrap_statistics import (
    align_paired_by_user,
    bootstrap_global_distributions,
    bootstrap_index_sha256,
    bootstrap_mean_ci,
    catalog_coverage,
    exposure_gini,
    generate_bootstrap_indices,
    paired_mean_delta_ci,
    percentile_interval,
    sample_mean_std,
    validate_bootstrap_indices,
)


def test_known_mean_and_sample_standard_deviation_use_ddof_one() -> None:
    mean, sample_std = sample_mean_std([1.0, 2.0, 3.0, 4.0])
    assert mean == 2.5
    assert sample_std == pytest.approx(math.sqrt(5.0 / 3.0), rel=0, abs=1e-15)
    assert sample_std != np.std([1.0, 2.0, 3.0, 4.0], ddof=0)


def test_constant_zero_vector_has_degenerate_interval_without_nan() -> None:
    indices = generate_bootstrap_indices(4, 1000, 42)
    interval = bootstrap_mean_ci([0.0, 0.0, 0.0, 0.0], indices)
    assert interval.observed_mean == 0.0
    assert interval.sample_std == 0.0
    assert (interval.ci_low, interval.ci_high) == (0.0, 0.0)
    delta = paired_mean_delta_ci([0.0] * 4, [0.0] * 4, indices)
    assert (delta.observed_delta, delta.ci_low, delta.ci_high) == (0.0, 0.0, 0.0)


def test_bootstrap_indices_have_frozen_shape_bounds_dtype_and_bytes() -> None:
    indices = generate_bootstrap_indices(7, 1000, 42)
    assert indices.shape == (1000, 7)
    assert indices.dtype == np.dtype("<i8")
    assert indices.flags.c_contiguous
    assert int(indices.min()) >= 0
    assert int(indices.max()) < 7
    expected = hashlib.sha256(
        np.asarray(indices, dtype="<i8", order="C").tobytes(order="C")
    ).hexdigest()
    assert bootstrap_index_sha256(indices) == expected


def test_bootstrap_indices_are_seed_reproducible_and_seed_sensitive() -> None:
    first = generate_bootstrap_indices(8, 1000, 42)
    second = generate_bootstrap_indices(8, 1000, 42)
    changed = generate_bootstrap_indices(8, 1000, 43)
    assert np.array_equal(first, second)
    assert bootstrap_index_sha256(first) == bootstrap_index_sha256(second)
    assert bootstrap_index_sha256(first) != bootstrap_index_sha256(changed)


@pytest.mark.parametrize("value", [True, 0, -1, 1.5, "1000"])
def test_invalid_resample_count_fails_closed(value) -> None:
    expected = TypeError if isinstance(value, (bool, float, str)) else ValueError
    with pytest.raises(expected):
        generate_bootstrap_indices(4, value, 42)


@pytest.mark.parametrize("seed", [True, -1, 1.5, "42"])
def test_invalid_seed_fails_closed(seed) -> None:
    expected = TypeError if seed != -1 else ValueError
    with pytest.raises(expected):
        generate_bootstrap_indices(4, 10, seed)


@pytest.mark.parametrize(
    "values",
    [[], [[1.0, 2.0]], [1.0, np.nan], [1.0, np.inf], [1.0, -np.inf]],
)
def test_invalid_metric_vectors_fail_closed(values) -> None:
    with pytest.raises(ValueError):
        sample_mean_std(values)


def test_bootstrap_index_validation_rejects_dtype_shape_count_and_bounds() -> None:
    valid = np.zeros((3, 2), dtype=np.int64)
    assert validate_bootstrap_indices(valid, n_observations=2, n_resamples=3).shape == (
        3,
        2,
    )
    with pytest.raises(TypeError, match="integer"):
        validate_bootstrap_indices(valid.astype(float), n_observations=2)
    with pytest.raises(ValueError, match="two-dimensional"):
        validate_bootstrap_indices(np.zeros(2, dtype=int), n_observations=2)
    with pytest.raises(ValueError, match="resample count"):
        validate_bootstrap_indices(valid, n_observations=2, n_resamples=4)
    with pytest.raises(ValueError, match="out of range"):
        validate_bootstrap_indices(np.array([[0, 2]]), n_observations=2)


def test_linear_percentile_interval_matches_manual_interpolation_fixture() -> None:
    low, high = percentile_interval([0.0, 1.0, 2.0, 100.0])
    assert low == pytest.approx(0.07500000000000007, rel=0, abs=1e-15)
    assert high == pytest.approx(92.65, rel=0, abs=5e-14)
    assert (low, high) != (0.0, 100.0)


def test_percentile_interval_rejects_wrong_method_and_confidence() -> None:
    with pytest.raises(ValueError, match="linear"):
        percentile_interval([1.0, 2.0], method="nearest")
    for confidence in (0.0, 1.0, np.nan, np.inf):
        with pytest.raises(ValueError):
            percentile_interval([1.0, 2.0], confidence_level=confidence)


def test_paired_bootstrap_reuses_same_indices() -> None:
    base = np.array([0.0, 100.0, 200.0, 300.0])
    comparison = base + 1.0
    indices = generate_bootstrap_indices(4, 1000, 42)
    interval = paired_mean_delta_ci(base, comparison, indices)
    assert interval.observed_delta == 1.0
    assert interval.ci_low == 1.0
    assert interval.ci_high == 1.0


def test_user_id_alignment_is_order_invariant() -> None:
    users, base, comparison = align_paired_by_user(
        [3, 1, 2],
        [30.0, 10.0, 20.0],
        [2, 3, 1],
        [21.0, 31.0, 11.0],
    )
    assert users.tolist() == [1, 2, 3]
    assert base.tolist() == [10.0, 20.0, 30.0]
    assert comparison.tolist() == [11.0, 21.0, 31.0]


@pytest.mark.parametrize(
    ("base_users", "comparison_users", "match"),
    [
        ([1, 1], [1, 2], "unique"),
        ([1, 2], [1, 3], "different user ID sets"),
        ([1, 2], [1, 2, 3], "different user ID sets"),
    ],
)
def test_user_id_alignment_rejects_duplicate_missing_and_extra_users(
    base_users,
    comparison_users,
    match,
) -> None:
    with pytest.raises(ValueError, match=match):
        align_paired_by_user(
            base_users,
            np.arange(len(base_users), dtype=float),
            comparison_users,
            np.arange(len(comparison_users), dtype=float),
        )


def test_coverage_uses_fixed_catalog_denominator() -> None:
    recommendations = np.array([[1, 2], [2, 3]], dtype=np.int64)
    assert catalog_coverage(recommendations, catalog_size=10) == 0.3
    assert catalog_coverage(recommendations, catalog_size=10) != 1.0


def test_gini_includes_zero_exposure_catalog_items() -> None:
    recommendations = np.array([[1], [2]], dtype=np.int64)
    assert exposure_gini(recommendations, (1, 2, 3, 4)) == 0.5
    assert exposure_gini(recommendations, (1, 2)) == 0.0


def test_global_bootstrap_preserves_repeated_user_multiplicity() -> None:
    recommendations = np.array([[1], [2], [3]], dtype=np.int64)
    indices = np.array([[0, 0, 1]], dtype=np.int64)
    coverage, gini = bootstrap_global_distributions(
        recommendations,
        indices,
        catalog_item_ids=(1, 2, 3, 4),
    )
    assert coverage.tolist() == [0.5]
    assert gini.tolist() == [pytest.approx(7.0 / 12.0)]
    assert gini[0] != 0.5


def test_global_metrics_reject_off_catalog_and_duplicate_recommendations() -> None:
    with pytest.raises(ValueError, match="absent"):
        exposure_gini(np.array([[1, 9]]), (1, 2, 3))
    with pytest.raises(ValueError, match="unique"):
        catalog_coverage(np.array([[1, 1]]), catalog_size=3)
