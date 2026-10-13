import json
from pathlib import Path

import numpy as np
import pytest

from recommender_core import ArtifactError, recommend_new_user
from recommender_core.local_artifact import FILES, load_local_bundle, write_local_bundle


@pytest.fixture
def bundle_path(tmp_path: Path) -> Path:
    return write_local_bundle(
        tmp_path / "artifact",
        item_ids=np.arange(1, 161, dtype=np.int64),
        item_factors=np.random.default_rng(7).normal(size=(160, 32)).astype(np.float32),
        popularity_counts=np.arange(1, 161, dtype=np.int64),
        catalog=[
            {
                "movie_id": i,
                "title": f"Fictional Movie {i} (2020)",
                "genres": ["Drama", "Comedy" if i % 2 else "Mystery"],
            }
            for i in range(1, 161)
        ],
        provenance={"source": "synthetic-test", "training_matrix_sha256": "a" * 64},
    )


def test_item_only_inventory_and_deterministic_exclusion(bundle_path):
    assert {p.name for p in bundle_path.iterdir()} == FILES
    bundle = load_local_bundle(bundle_path)
    assert bundle.existing_user_ids() == ()
    kwargs = {"configuration_key": "A1_L07", "excluded_movie_ids": [6, 7]}
    first = recommend_new_user(bundle, [1, 2, 3, 4, 5], **kwargs)
    assert first == recommend_new_user(
        load_local_bundle(bundle_path), [1, 2, 3, 4, 5], **kwargs
    )
    assert len(first) == 10
    assert not {r.movie_id for r in first} & set(range(1, 8))


@pytest.mark.parametrize(
    "mutation", ["checksum", "extra", "symlink", "versions", "identity"]
)
def test_mutated_bundle_fails_closed(bundle_path, mutation):
    if mutation == "checksum":
        with (bundle_path / "item_factors.npy").open("ab") as stream:
            stream.write(b"corruption")
    elif mutation == "extra":
        (bundle_path / "user_ids.npy").write_bytes(b"unapproved payload")
    elif mutation == "symlink":
        target = bundle_path / "catalog.json"
        alternate = bundle_path.parent / "catalog-copy.json"
        target.rename(alternate)
        target.symlink_to(alternate)
    else:
        manifest = json.loads((bundle_path / "manifest.json").read_text())
        if mutation == "versions":
            manifest["numerical_versions"]["numpy"] = "incompatible"
        else:
            manifest["bundle_sha256"] = "0" * 64
        (bundle_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ArtifactError):
        load_local_bundle(bundle_path)
