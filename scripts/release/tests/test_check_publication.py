from __future__ import annotations

import importlib.util
import subprocess
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "check_publication.py"
SPEC = importlib.util.spec_from_file_location("check_publication", SCRIPT)
assert SPEC and SPEC.loader
subject = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(subject)


class PublicationTests(unittest.TestCase):
    def test_examples_and_authored_source_are_allowed(self):
        for path in (
            ".env.example",
            ".env.local.example",
            "backend/app/main.py",
            "data/processed/data_manifest.json",
            "reports/tables/exp003_model_comparison_validation.csv",
        ):
            with self.subTest(path=path):
                self.assertIsNone(subject.category(path))

    def test_known_dataset_secrets_and_runtime_paths_are_blocked(self):
        for path in (
            "frontend/.env.production",
            ".env",
            ".local/auth-secret",
            "artifacts/serving/model/seen_indices.npy",
            "artifacts/models/model/user_factors.npy",
            "data/raw/ml-1m.zip",
            "data/processed/train.parquet",
            "reports/baselines/model_per_user.csv",
            "reports/reranking/lambda_sweep_candidate_manifest.csv",
            "backup.dump",
        ):
            with self.subTest(path=path):
                self.assertIsNotNone(subject.category(path))

    def test_deleted_payload_remains_blocked_in_history(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)

            def git(*args):
                return subprocess.run(
                    ["git", "-C", str(root), *args],
                    check=True,
                    capture_output=True,
                    text=True,
                )

            git("init")
            git("config", "user.name", "Fixture")
            git("config", "user.email", "fixture@example.com")
            (root / "README.md").write_text("fixture")
            (root / ".env.production").write_text("synthetic fixture")
            git("add", ".")
            git("commit", "-m", "fixture with restricted path")
            (root / ".env.production").unlink()
            git("add", "-u")
            git("commit", "-m", "remove from current tree")
            self.assertEqual(subject.scan(root, tree_only=True)["findings"], [])
            self.assertEqual(
                subject.scan(root)["findings"],
                [{"path": ".env.production", "category": "environment_secret_file"}],
            )

    def test_non_head_branch_is_checked(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)

            def git(*args):
                return subprocess.run(
                    ["git", "-C", str(root), *args],
                    check=True,
                    capture_output=True,
                    text=True,
                )

            git("init")
            git("config", "user.name", "Fixture")
            git("config", "user.email", "fixture@example.com")
            (root / "README.md").write_text("fixture")
            git("add", ".")
            git("commit", "-m", "clean base")
            git("branch", "clean")
            git("checkout", "-b", "restricted")
            (root / "users.dat").write_text("synthetic fixture")
            git("add", ".")
            git("commit", "-m", "restricted branch")
            git("checkout", "clean")
            self.assertEqual(subject.scan(root, tree_only=True)["findings"], [])
            self.assertTrue(
                any(x["path"] == "users.dat" for x in subject.scan(root)["findings"])
            )

    def test_shallow_checkout_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            subprocess.run(["git", "init", str(root)], check=True, capture_output=True)
            (root / ".git/shallow").write_text("0" * 40 + "\n")
            with self.assertRaises(ValueError):
                subject.scan(root)


if __name__ == "__main__":
    unittest.main()
