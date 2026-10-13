import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from urllib.request import urlretrieve

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

MANIFEST_PATH = PROCESSED_DATA_DIR / "data_manifest.json"

DATASET_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"

README_URL = "https://files.grouplens.org/datasets/movielens/ml-1m-README.txt"

EXPECTED_SHA256 = "a6898adb50b9ca05aa231689da44c217cb524e7ebd39d264c56e2832f2c54e20"

ZIP_PATH = RAW_DATA_DIR / "ml-1m.zip"
README_PATH = RAW_DATA_DIR / "ml-1m-README.txt"


def calculate_sha256(file_path: Path) -> str:
    """
    Calculate the SHA-256 digest of a file.

    The file is read in one-megabyte chunks so the complete file
    does not need to be loaded into memory.
    """
    sha256 = hashlib.sha256()

    with file_path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            sha256.update(chunk)

    return sha256.hexdigest()


def validate_readme(readme_path: Path) -> None:
    """Validate important GroupLens usage-license phrases."""
    content = readme_path.read_text(
        encoding="utf-8",
        errors="replace",
    )

    normalized_content = content.lower()

    # Repair words that were split across lines after a hyphen.
    normalized_content = re.sub(
        r"-\s+",
        "-",
        normalized_content,
    )

    normalized_content = " ".join(normalized_content.split())

    required_phrases = [
        "usage license",
        "may not redistribute",
        "commercial or revenue-bearing",
        "citation",
    ]

    for phrase in required_phrases:
        if phrase not in normalized_content:
            raise RuntimeError(f"Required phrase not found in README: {phrase}")

    print("README license validation passed.")


def download_dataset() -> str:
    """
    Download and verify the MovieLens 1M archive.

    An existing archive is reused only when its SHA-256 digest is valid.
    The verified local SHA-256 value is returned.
    """
    if ZIP_PATH.exists():
        actual_sha256 = calculate_sha256(ZIP_PATH)

        print("Dataset already exists.")
        print("Actual SHA-256:", actual_sha256)

        if actual_sha256 == EXPECTED_SHA256:
            print("SHA-256 verification passed.")
            return actual_sha256

        print("Existing dataset file is invalid.")
        print("Deleting invalid file...")

        ZIP_PATH.unlink()

    print("Downloading MovieLens 1M from GroupLens...")

    temporary_path = ZIP_PATH.with_suffix(".zip.part")

    try:
        urlretrieve(DATASET_URL, temporary_path)
        temporary_path.replace(ZIP_PATH)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise

    print("Download completed.")
    print("Downloaded file:", ZIP_PATH)

    actual_sha256 = calculate_sha256(ZIP_PATH)

    print("Expected SHA-256:", EXPECTED_SHA256)
    print("Actual SHA-256:  ", actual_sha256)

    if actual_sha256 != EXPECTED_SHA256:
        ZIP_PATH.unlink(missing_ok=True)

        raise RuntimeError("Downloaded dataset SHA-256 does not match.")

    print("SHA-256 verification passed.")

    return actual_sha256


def download_readme() -> None:
    """Download and validate the official GroupLens README."""
    if README_PATH.exists():
        print("Official README already exists.")
    else:
        print("Downloading official GroupLens README...")

        temporary_path = README_PATH.with_suffix(".txt.part")

        try:
            urlretrieve(README_URL, temporary_path)
            temporary_path.replace(README_PATH)
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise

        print("README download completed.")
        print("README path:", README_PATH)

    validate_readme(README_PATH)


def load_existing_manifest() -> dict[str, object]:
    """
    Load the existing manifest.

    Existing sections created by later pipeline stages are preserved.
    """
    if not MANIFEST_PATH.exists():
        return {}

    try:
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise RuntimeError(
            f"Existing data manifest is invalid JSON: {error}"
        ) from error

    if not isinstance(manifest, dict):
        raise TypeError("Existing data manifest must contain a JSON object")

    return manifest


def update_manifest_section(
    manifest: dict[str, object],
    section_name: str,
    values: dict[str, object],
) -> None:
    """
    Update one manifest section without deleting unknown existing fields.
    """
    existing_section = manifest.get(section_name)

    if isinstance(existing_section, dict):
        section = existing_section.copy()
    else:
        section = {}

    section.update(values)
    manifest[section_name] = section


def write_manifest(archive_sha256: str) -> None:
    """
    Update download metadata while preserving later pipeline metadata.
    """
    PROCESSED_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest = load_existing_manifest()

    readme_sha256 = calculate_sha256(README_PATH)
    now_utc = datetime.now(UTC).isoformat()

    existing_download = manifest.get("download")

    if isinstance(existing_download, dict):
        existing_verified_at = existing_download.get("verified_at_utc")
    else:
        existing_verified_at = None

    verified_at_utc = (
        existing_verified_at if isinstance(existing_verified_at, str) else now_utc
    )
    manifest["manifest_schema_version"] = 1

    update_manifest_section(
        manifest,
        "dataset",
        {
            "name": "MovieLens 1M",
            "version": "1m",
            "provider": "GroupLens Research",
        },
    )

    update_manifest_section(
        manifest,
        "source",
        {
            "archive_url": DATASET_URL,
            "readme_url": README_URL,
        },
    )

    update_manifest_section(
        manifest,
        "download",
        {
            "archive_path": "data/raw/ml-1m.zip",
            "archive_size_bytes": ZIP_PATH.stat().st_size,
            "expected_sha256": EXPECTED_SHA256,
            "actual_sha256": archive_sha256,
            "sha256_verified": archive_sha256 == EXPECTED_SHA256,
            "verified_at_utc": verified_at_utc,
        },
    )

    update_manifest_section(
        manifest,
        "license",
        {
            "official_readme_path": "data/raw/ml-1m-README.txt",
            "official_readme_sha256": readme_sha256,
            "citation_required": True,
            "raw_data_redistribution_allowed": False,
            "commercial_use_requires_permission": True,
        },
    )

    if "created_at_utc" not in manifest:
        manifest["created_at_utc"] = verified_at_utc

    if "updated_at_utc" not in manifest:
        manifest["updated_at_utc"] = verified_at_utc

    serialized_manifest = (
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )

    temporary_path = MANIFEST_PATH.with_suffix(".json.tmp")

    try:
        temporary_path.write_text(
            serialized_manifest,
            encoding="utf-8",
        )
        temporary_path.replace(MANIFEST_PATH)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise

    print("Manifest written:", MANIFEST_PATH)


def main() -> None:
    """Run the MovieLens download and verification pipeline."""
    RAW_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Project root:", PROJECT_ROOT)
    print("Dataset URL:", DATASET_URL)
    print("Download path:", ZIP_PATH)

    archive_sha256 = download_dataset()

    download_readme()
    write_manifest(archive_sha256)

    print("Dataset download and verification completed successfully.")


if __name__ == "__main__":
    main()
