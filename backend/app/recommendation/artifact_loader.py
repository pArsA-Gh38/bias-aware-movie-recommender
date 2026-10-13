"""Backend boundary for loading the authoritative ART-002 serving bundle."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from recommender_core import (
    ArtifactError,
    ServingBundleV2,
    ServingRecommendation,
    load_serving_bundle_v2,
    recommend_new_user,
)
from recommender_core.local_artifact import FORMAT, load_local_bundle


class ArtifactUnavailableError(ArtifactError):
    """The configured serving artifact does not exist."""


@dataclass(slots=True)
class ArtifactLoader:
    """Load ART-002 through the public recommender-core contract exactly once."""

    path: Path
    bundle: ServingBundleV2 | None = field(default=None, init=False)
    error: ArtifactError | None = field(default=None, init=False)

    def load(self) -> ServingBundleV2:
        """Validate and retain the configured v2 bundle, or fail closed."""
        self.bundle = None
        self.error = None
        if not self.path.is_dir():
            error = ArtifactUnavailableError(
                f"configured ART-002 bundle is unavailable: {self.path}"
            )
            self.error = error
            raise error
        try:
            manifest = json.loads((self.path / "manifest.json").read_text())
            bundle = (
                load_local_bundle(self.path)
                if manifest.get("format") == FORMAT
                else load_serving_bundle_v2(self.path)
            )
        except ArtifactError as error:
            self.error = error
            raise
        except (OSError, ValueError, TypeError) as cause:
            error = ArtifactError("invalid local serving manifest")
            self.error = error
            raise error from cause
        self.bundle = bundle
        return bundle

    def recommend_new_user(
        self,
        positive_movie_ids: tuple[int, ...],
        configuration_key: str,
        *,
        excluded_movie_ids: tuple[int, ...],
    ) -> tuple[ServingRecommendation, ...]:
        """Run fold-in through the additive public core serving API."""
        if self.bundle is None:
            raise ArtifactUnavailableError("ART-002 is not loaded")
        return recommend_new_user(
            self.bundle,
            positive_movie_ids,
            configuration_key,
            excluded_movie_ids=excluded_movie_ids,
        )

    @property
    def is_loaded(self) -> bool:
        return self.bundle is not None

    @property
    def schema_compatible(self) -> bool | None:
        if self.bundle is not None:
            return True
        if self.error is None or isinstance(self.error, ArtifactUnavailableError):
            return None
        return False
