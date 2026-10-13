#!/usr/bin/env python3
"""Fail-closed AST/import audit for the ARCH-001 runtime boundary."""

from __future__ import annotations

import ast
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CORE = ROOT / "packages/recommender_core/src/recommender_core"
ML = ROOT / "ml/src"
ML_SCRIPTS = ROOT / "ml/scripts"
BACKEND_APP = ROOT / "backend/app"
BACKEND_TESTS = ROOT / "backend/tests"
COMPATIBILITY_MODULES = (
    ML / "inference/als_fold_in.py",
    ML / "reranking/normalization.py",
    ML / "reranking/greedy_reranker.py",
)
MOVED_FUNCTIONS = {
    "build_fold_in_confidence_row",
    "decide_fold_in_strategy",
    "fold_in_new_user",
    "greedy_rerank",
    "load_serving_bundle",
    "min_max_normalize_candidate_feature",
    "normalize_reranking_features",
    "rank_factor_items",
}


def _python_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*.py") if path.is_file())


def _imports(tree: ast.AST) -> set[str]:
    roots = {
        alias.name.split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    roots.update(
        node.module.split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    )
    return roots


def _is_algorithmic(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    return any(
        isinstance(
            descendant,
            (
                ast.For,
                ast.While,
                ast.ListComp,
                ast.SetComp,
                ast.DictComp,
                ast.GeneratorExp,
                ast.MatMult,
            ),
        )
        for descendant in ast.walk(node)
    )


def _fingerprint(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    normalized = ast.FunctionDef(
        name="_",
        args=node.args,
        body=node.body,
        decorator_list=[],
        returns=node.returns,
        type_comment=node.type_comment,
        type_params=getattr(node, "type_params", []),
    )
    return hashlib.sha256(
        ast.dump(normalized, include_attributes=False).encode()
    ).hexdigest()


def audit() -> list[str]:
    errors: list[str] = []
    trees: dict[Path, ast.Module] = {}
    for root in (CORE, ML, ML_SCRIPTS, BACKEND_APP, BACKEND_TESTS):
        for path in _python_files(root):
            try:
                trees[path] = ast.parse(path.read_text(encoding="utf-8"))
            except (OSError, SyntaxError) as error:
                errors.append(f"cannot parse {path.relative_to(ROOT)}: {error}")

    for path, tree in trees.items():
        imports = _imports(tree)
        if path.is_relative_to(CORE) and imports & {"ml", "backend", "src"}:
            errors.append(f"core reverse dependency: {path.relative_to(ROOT)}")
        if (
            path.is_relative_to(BACKEND_APP) or path.is_relative_to(BACKEND_TESTS)
        ) and imports & {"ml", "src"}:
            errors.append(f"backend private ML dependency: {path.relative_to(ROOT)}")

    ml_imports = set().union(
        *(
            _imports(tree)
            for path, tree in trees.items()
            if path.is_relative_to(ML) or path.is_relative_to(ML_SCRIPTS)
        )
    )
    backend_imports = set().union(
        *(
            _imports(tree)
            for path, tree in trees.items()
            if path.is_relative_to(BACKEND_APP) or path.is_relative_to(BACKEND_TESTS)
        )
    )
    if "recommender_core" not in ml_imports:
        errors.append("missing ml -> recommender_core dependency edge")
    if "recommender_core" not in backend_imports:
        errors.append("missing backend -> recommender_core dependency edge")

    for path in COMPATIBILITY_MODULES:
        tree = trees.get(path)
        if tree is None:
            errors.append(f"missing compatibility module: {path.relative_to(ROOT)}")
            continue
        forbidden_nodes = (
            ast.FunctionDef,
            ast.AsyncFunctionDef,
            ast.ClassDef,
            ast.For,
            ast.While,
            ast.If,
            ast.Try,
            ast.With,
        )
        if any(isinstance(node, forbidden_nodes) for node in ast.walk(tree)):
            errors.append(f"algorithmic compatibility module: {path.relative_to(ROOT)}")

    definitions: dict[
        str, list[tuple[Path, ast.FunctionDef | ast.AsyncFunctionDef]]
    ] = {}
    fingerprints: dict[str, list[tuple[Path, str]]] = {}
    for path, tree in trees.items():
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                definitions.setdefault(node.name, []).append((path, node))
                if _is_algorithmic(node):
                    fingerprints.setdefault(_fingerprint(node), []).append(
                        (path, node.name)
                    )

    for name in MOVED_FUNCTIONS:
        core_definitions = [
            path for path, _ in definitions.get(name, []) if path.is_relative_to(CORE)
        ]
        if len(core_definitions) != 1:
            errors.append(
                f"moved function {name} has {len(core_definitions)} core owners"
            )
        for path, node in definitions.get(name, []):
            if path.is_relative_to(CORE):
                continue
            if name == "fold_in_new_user" and path == ML / "artifacts/als_artifact.py":
                forbidden_calls = {
                    child.func.attr
                    for child in ast.walk(node)
                    if isinstance(child, ast.Call)
                    and isinstance(child.func, ast.Attribute)
                }
                if (
                    any(isinstance(child, ast.MatMult) for child in ast.walk(node))
                    or {
                        "sort",
                        "recalculate_user",
                    }
                    & forbidden_calls
                ):
                    errors.append(
                        "ML artifact fold-in adapter contains moved algorithm structure"
                    )
                continue
            errors.append(f"duplicate moved function {name}: {path.relative_to(ROOT)}")

    for entries in fingerprints.values():
        owners = {
            "core" if path.is_relative_to(CORE) else "outside" for path, _ in entries
        }
        if owners == {"core", "outside"}:
            rendered = ", ".join(
                f"{path.relative_to(ROOT)}:{name}" for path, name in entries
            )
            errors.append(f"duplicate algorithm AST fingerprint: {rendered}")

    artifact_tree = trees.get(ML / "artifacts/als_artifact.py")
    if artifact_tree is not None:
        for node in ast.walk(artifact_tree):
            if (
                isinstance(node, ast.FunctionDef)
                and node.name == "recommend_existing_user"
            ):
                attribute_calls = {
                    child.func.attr
                    for child in ast.walk(node)
                    if isinstance(child, ast.Call)
                    and isinstance(child.func, ast.Attribute)
                }
                if (
                    any(isinstance(child, ast.MatMult) for child in ast.walk(node))
                    or "sort" in attribute_calls
                ):
                    errors.append(
                        "ML existing-user adapter retains scoring/ranking algorithm"
                    )
                calls = {
                    child.func.id
                    for child in ast.walk(node)
                    if isinstance(child, ast.Call) and isinstance(child.func, ast.Name)
                }
                if "rank_factor_items" not in calls:
                    errors.append("ML existing-user adapter does not delegate to core")
    return errors


def main() -> int:
    errors = audit()
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print("PASS: runtime dependency direction and duplicate-implementation audit")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
