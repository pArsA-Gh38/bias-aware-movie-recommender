"""Reject sensitive payload paths in the working tree's index or fetched history.

This is a publication inventory guard, not a secret-content scanner or legal clearance.
It prints paths and categories only; it never prints file contents.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

DATA_NAMES = {
    "ratings.dat",
    "users.dat",
    "movies.dat",
    "ml-1m.zip",
    "seen_indices.npy",
    "seen_indptr.npy",
    "user_ids.npy",
    "user_factors.npy",
    "fit_positive_counts.npy",
}
DATA_REPORT = re.compile(
    r"^reports/.*(?:_per_user|_recommendations|_candidate_manifest)\.csv$"
)


def category(path: str) -> str | None:
    parts = Path(path).parts
    name = Path(path).name
    if name in DATA_NAMES or DATA_REPORT.fullmatch(path):
        return "dataset_or_user_payload"
    if Path(path).suffix.lower() in {
        ".npy",
        ".npz",
        ".pkl",
        ".pickle",
        ".parquet",
        ".pt",
        ".pth",
    }:
        return "generated_model_or_dataset_payload"
    if parts[:2] in (("data", "raw"), ("data", "processed")) and name not in {
        ".gitkeep",
        "data_manifest.json",
    }:
        return "dataset_payload"
    if any(part in {".local", "node_modules", ".next", ".venv"} for part in parts):
        return "runtime_or_dependency_state"
    if name == ".env" or (name.startswith(".env.") and not name.endswith(".example")):
        return "environment_secret_file"
    if Path(path).suffix.lower() in {
        ".key",
        ".pem",
        ".p12",
        ".pfx",
        ".sqlite",
        ".db",
        ".dump",
        ".age",
    }:
        return "credential_or_database_backup"
    return None


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), *args], text=True, stderr=subprocess.PIPE
    )


def scan(root: Path, *, tree_only: bool = False) -> dict[str, object]:
    shallow = git(root, "rev-parse", "--is-shallow-repository").strip() == "true"
    if not tree_only and shallow:
        raise ValueError(
            "full-history publication checks require a non-shallow checkout"
        )
    if tree_only:
        paths = set(git(root, "ls-files", "-z").rstrip("\0").split("\0"))
    else:
        # Uses all locally available refs, not just HEAD; deleted paths remain visible.
        paths = {
            row.partition(" ")[2]
            for row in git(root, "rev-list", "--objects", "--all").splitlines()
            if " " in row
        }
    findings = [
        {"path": path, "category": kind}
        for path in sorted(paths)
        if path and (kind := category(path))
    ]
    return {
        "scope": "tracked_tree" if tree_only else "all_locally_fetched_history",
        "paths_checked": len(paths),
        "findings": findings,
        "status": "BLOCKED" if findings else "PATH_CHECK_PASS",
        "limitations": "Does not scan secret contents, remote-only refs, releases, packages or attachments.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    parser.add_argument("--tree-only", action="store_true")
    args = parser.parse_args()
    try:
        result = scan(args.root, tree_only=args.tree_only)
    except (OSError, subprocess.CalledProcessError, ValueError) as error:
        # Do not expose credential-bearing command output or stderr.
        print(json.dumps({"status": "ERROR", "error_type": type(error).__name__}))
        return 2
    print(json.dumps(result, indent=2))
    return 1 if result["findings"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
