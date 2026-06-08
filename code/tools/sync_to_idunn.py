#!/usr/bin/env python3
"""Sync the publication subset of idunn_dev to a sibling idunn checkout.

Reads `.idunn-sync.yaml` at the idunn_dev repo root for classification rules
(see ROADMAP item 116). Dry-run by default; no writes happen without --apply.

Typical use:

    # Preview what would change
    python code/tools/sync_to_idunn.py --idunn-path ../idunn

    # Show per-file diffs for files that differ
    python code/tools/sync_to_idunn.py --idunn-path ../idunn --verbose

    # Apply changes on a topic branch in the idunn checkout (no commit)
    python code/tools/sync_to_idunn.py --idunn-path ../idunn \\
        --apply --branch sync/$(date +%Y%m%d)-pre-lock

Exit codes:
    0  in-sync (nothing to copy / delete / patch)
    1  sync needed (or, with --apply, sync completed and changes are staged)
    2  configuration / invocation error
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

try:
    import yaml
except ImportError:
    sys.stderr.write(
        "ERROR: PyYAML is required. Activate balder_conda_env or `pip install pyyaml`.\n"
    )
    sys.exit(2)


CONFIG_FILENAME = ".idunn-sync.yaml"


# ---------------------------------------------------------------------------
# Glob matching
# ---------------------------------------------------------------------------

def _glob_to_regex(pattern: str) -> re.Pattern:
    """Translate a sync-yaml glob to a regex.

    Semantics (subset of gitignore / fnmatch):
      **/  -> any number of directory segments (including zero)
      /**  -> any subtree under this directory
      **   -> any path
      *    -> any chars except /
      ?    -> single char except /
    """
    p = pattern.replace("\\", "/")
    out: list[str] = []
    i = 0
    while i < len(p):
        if p[i:i + 3] == "**/":
            out.append("(?:.*/)?")
            i += 3
        elif p[i:i + 3] == "/**":
            out.append("(?:/.*)?")
            i += 3
        elif p[i:i + 2] == "**":
            out.append(".*")
            i += 2
        elif p[i] == "*":
            out.append("[^/]*")
            i += 1
        elif p[i] == "?":
            out.append("[^/]")
            i += 1
        elif p[i] in r".+()|^$[]{}\\":
            out.append("\\" + p[i])
            i += 1
        else:
            out.append(p[i])
            i += 1
    return re.compile("^" + "".join(out) + "$")


def _matches_any(path: str, patterns: list[re.Pattern]) -> bool:
    return any(pat.match(path) for pat in patterns)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class SyncConfig:
    include: list[re.Pattern] = field(default_factory=list)
    exclude: list[re.Pattern] = field(default_factory=list)
    idunn_only: set[str] = field(default_factory=set)

    @classmethod
    def load(cls, path: Path) -> "SyncConfig":
        data = yaml.safe_load(path.read_text()) or {}
        return cls(
            include=[_glob_to_regex(p) for p in data.get("include", [])],
            exclude=[_glob_to_regex(p) for p in data.get("exclude", [])],
            idunn_only=set(data.get("idunn_only", [])),
        )

    def file_in_scope(self, rel_path: str) -> bool:
        return _matches_any(rel_path, self.include) and not _matches_any(
            rel_path, self.exclude
        )


# ---------------------------------------------------------------------------
# Filesystem walk
# ---------------------------------------------------------------------------

def _walk_repo(root: Path) -> list[str]:
    """Return sorted list of relative POSIX paths for all files under root.

    Skips __pycache__ and any path component starting with '.' (e.g. .git,
    .pytest_cache, .claude, .gitignore). This makes repo-infrastructure files
    invisible to the sync on both sides, so they aren't ever flagged for
    copy or delete.
    """
    files: list[str] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        parts = rel.split("/")
        if any(part == "__pycache__" or part.startswith(".") for part in parts):
            continue
        files.append(rel)
    files.sort()
    return files


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Diff bucketing
# ---------------------------------------------------------------------------

@dataclass
class SyncPlan:
    to_copy: list[str] = field(default_factory=list)        # A-only
    to_delete: list[str] = field(default_factory=list)      # B-only, not protected
    diverged: list[str] = field(default_factory=list)       # in both, sha differs
    in_sync: list[str] = field(default_factory=list)        # in both, sha matches
    protected: list[str] = field(default_factory=list)      # idunn_only present
    out_of_scope_in_dev: list[str] = field(default_factory=list)  # informational

    def needs_sync(self) -> bool:
        return bool(self.to_copy or self.to_delete or self.diverged)


def build_plan(
    dev_root: Path, idunn_root: Path, cfg: SyncConfig
) -> SyncPlan:
    dev_files_all = _walk_repo(dev_root)
    idunn_files_all = _walk_repo(idunn_root)

    dev_in_scope: set[str] = set()
    for f in dev_files_all:
        if cfg.file_in_scope(f):
            dev_in_scope.add(f)

    idunn_present = set(idunn_files_all)

    plan = SyncPlan()
    plan.out_of_scope_in_dev = sorted(set(dev_files_all) - dev_in_scope)

    for f in sorted(dev_in_scope):
        if f in idunn_present:
            if _sha256(dev_root / f) == _sha256(idunn_root / f):
                plan.in_sync.append(f)
            else:
                plan.diverged.append(f)
        else:
            plan.to_copy.append(f)

    for f in sorted(idunn_present):
        if f in dev_in_scope:
            continue
        if f in cfg.idunn_only:
            plan.protected.append(f)
        else:
            plan.to_delete.append(f)

    return plan


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _print_section(title: str, items: list[str], limit: int | None = None) -> None:
    if not items:
        return
    print(f"\n{title} ({len(items)}):")
    shown = items if limit is None else items[:limit]
    for item in shown:
        print(f"  {item}")
    if limit is not None and len(items) > limit:
        print(f"  ... and {len(items) - limit} more (use --verbose to see all)")


def _print_diff(dev_path: Path, idunn_path: Path, rel: str, context: int = 3) -> None:
    try:
        dev_lines = dev_path.read_text().splitlines(keepends=True)
        idunn_lines = idunn_path.read_text().splitlines(keepends=True)
    except UnicodeDecodeError:
        print(f"  (binary file, diff omitted)")
        return
    diff = difflib.unified_diff(
        idunn_lines, dev_lines,
        fromfile=f"idunn/{rel}", tofile=f"idunn_dev/{rel}",
        n=context,
    )
    sys.stdout.writelines(diff)


def print_report(
    plan: SyncPlan, dev_root: Path, idunn_root: Path, verbose: bool
) -> None:
    print("Summary:")
    print(f"  In sync       : {len(plan.in_sync):>4} files")
    print(f"  To copy (A→B) : {len(plan.to_copy):>4} files")
    print(f"  To delete (B) : {len(plan.to_delete):>4} files")
    print(f"  Diverged      : {len(plan.diverged):>4} files")
    print(f"  Protected     : {len(plan.protected):>4} files (idunn_only)")

    limit = None if verbose else 25
    _print_section("To copy (idunn_dev → idunn)", plan.to_copy, limit)
    _print_section("To delete (from idunn)", plan.to_delete, limit)
    _print_section("Diverged (will overwrite idunn copy)", plan.diverged, limit)
    if verbose:
        _print_section("In sync", plan.in_sync, limit=None)
        _print_section("Protected (idunn_only)", plan.protected, limit=None)
        _print_section(
            "Out-of-scope in idunn_dev (not synced)",
            plan.out_of_scope_in_dev, limit=None,
        )

    if verbose and plan.diverged:
        print("\nDiverged file diffs (idunn → idunn_dev):")
        for rel in plan.diverged:
            print(f"\n--- {rel}")
            _print_diff(dev_root / rel, idunn_root / rel, rel)


# ---------------------------------------------------------------------------
# Apply
# ---------------------------------------------------------------------------

def _run(cmd: list[str], cwd: Path) -> str:
    result = subprocess.run(
        cmd, cwd=cwd, capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


def _ensure_clean_worktree(repo: Path) -> None:
    status = _run(["git", "status", "--porcelain"], cwd=repo)
    if status:
        raise SystemExit(
            f"ERROR: idunn working tree is dirty:\n{status}\n"
            "Commit or stash before --apply."
        )


def _create_branch(repo: Path, branch: str) -> None:
    existing = _run(["git", "branch", "--list", branch], cwd=repo)
    if existing:
        raise SystemExit(
            f"ERROR: branch '{branch}' already exists in {repo}. "
            "Pick a fresh name or delete it first."
        )
    _run(["git", "switch", "-c", branch], cwd=repo)


def apply_plan(
    plan: SyncPlan, dev_root: Path, idunn_root: Path, branch: str | None
) -> None:
    _ensure_clean_worktree(idunn_root)
    if branch:
        _create_branch(idunn_root, branch)
        print(f"Created and switched to branch: {branch}")

    # Copies (new files) and overwrites (diverged) are the same operation.
    for rel in plan.to_copy + plan.diverged:
        src = dev_root / rel
        dst = idunn_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    if plan.to_copy or plan.diverged:
        print(f"Copied {len(plan.to_copy) + len(plan.diverged)} files into idunn.")

    # Deletes; clean up any directories left empty.
    deleted_dirs: set[Path] = set()
    for rel in plan.to_delete:
        target = idunn_root / rel
        if target.exists():
            target.unlink()
        deleted_dirs.add(target.parent)
    for d in sorted(deleted_dirs, key=lambda p: len(p.parts), reverse=True):
        try:
            if d != idunn_root and not any(d.iterdir()):
                d.rmdir()
        except OSError:
            pass
    if plan.to_delete:
        print(f"Deleted {len(plan.to_delete)} files from idunn.")

    print(
        "\nChanges are staged in the idunn working tree but NOT committed.\n"
        f"Review with: cd {idunn_root} && git status && git diff\n"
        "Then commit and open a PR."
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _repo_root(start: Path) -> Path:
    """Locate the idunn_dev repo root by walking up for .idunn-sync.yaml."""
    cur = start.resolve()
    for parent in (cur, *cur.parents):
        if (parent / CONFIG_FILENAME).exists():
            return parent
    raise SystemExit(
        f"ERROR: could not find {CONFIG_FILENAME} in {start} or any parent."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--idunn-path", required=True, type=Path,
        help="Path to the sibling idunn checkout.",
    )
    parser.add_argument(
        "--config", type=Path, default=None,
        help=f"Path to sync config (default: {CONFIG_FILENAME} at repo root).",
    )
    parser.add_argument(
        "--apply", action="store_true",
        help="Apply changes (copies + deletes). Default is dry-run.",
    )
    parser.add_argument(
        "--branch", type=str, default=None,
        help="When --apply, create this branch in idunn before changes.",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Show full file lists and per-file diffs for diverged files.",
    )
    args = parser.parse_args()

    dev_root = _repo_root(Path(__file__).parent)
    idunn_root = args.idunn_path.resolve()
    if not idunn_root.is_dir():
        sys.stderr.write(f"ERROR: idunn path does not exist: {idunn_root}\n")
        return 2
    if not (idunn_root / ".git").exists():
        sys.stderr.write(f"ERROR: {idunn_root} is not a git repo.\n")
        return 2

    config_path = args.config or (dev_root / CONFIG_FILENAME)
    if not config_path.exists():
        sys.stderr.write(f"ERROR: config not found: {config_path}\n")
        return 2

    if args.branch and not args.apply:
        sys.stderr.write("ERROR: --branch only meaningful with --apply.\n")
        return 2

    print(f"idunn_dev : {dev_root}")
    print(f"idunn     : {idunn_root}")
    print(f"config    : {config_path}\n")

    cfg = SyncConfig.load(config_path)
    plan = build_plan(dev_root, idunn_root, cfg)
    print_report(plan, dev_root, idunn_root, verbose=args.verbose)

    if not plan.needs_sync():
        print("\nidunn is in sync with idunn_dev (per .idunn-sync.yaml). ✓")
        return 0

    if args.apply:
        print()
        apply_plan(plan, dev_root, idunn_root, args.branch)
        return 1  # changes were applied; exit non-zero so CI can detect drift

    print(
        "\nDry-run only. Re-run with --apply --branch sync/<topic> to apply.\n"
        "Exit code 1 = sync needed."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
