from pathlib import Path

import numpy as np
import pytest
from recommender_core.local_artifact import write_local_bundle


@pytest.fixture
def local_artifact(tmp_path: Path) -> Path:
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
