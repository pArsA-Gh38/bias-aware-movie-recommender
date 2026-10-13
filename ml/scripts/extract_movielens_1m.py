from pathlib import Path
from zipfile import ZipFile

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
ARCHIVE_PATH = REPOSITORY_ROOT / "data" / "raw" / "ml-1m.zip"
EXTRACT_DIRECTORY = REPOSITORY_ROOT / "data" / "raw"

EXPECTED_FILES = (
    "ml-1m/ratings.dat",
    "ml-1m/movies.dat",
    "ml-1m/users.dat",
    "ml-1m/README",
)


def extract_movielens() -> None:
    """Safely extract the MovieLens 1M archive into data/raw."""

    destination = EXTRACT_DIRECTORY.resolve()

    with ZipFile(ARCHIVE_PATH) as archive:
        for member in archive.infolist():
            target_path = (destination / member.filename).resolve()

            if not target_path.is_relative_to(destination):
                raise RuntimeError(f"Unsafe archive member path: {member.filename}")

        archive.extractall(destination)

    missing_files = [
        filename
        for filename in EXPECTED_FILES
        if not (destination / filename).is_file()
    ]

    if missing_files:
        raise RuntimeError(f"Expected extracted files are missing: {missing_files}")

    print("MovieLens 1M archive extracted successfully.")

    for filename in EXPECTED_FILES:
        print(f"- {filename}")


if __name__ == "__main__":
    extract_movielens()
