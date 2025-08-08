#!/usr/bin/env python3
"""
Safe housekeeping utility: move artifacts into backup/<timestamp>.
- Dry-run by default; use --apply to execute moves
- Never touches source files
- Optionally prunes old backups with --keep N (default 10)
"""
from __future__ import annotations

import argparse
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import List

ARTIFACT_PATTERNS: List[str] = [
    "logs",
    "results",
    "evaluation_results",
    "checkpoints",
    "wandb",
    "*.out",
    "*.err",
    "*.log",
]


def ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def resolve_paths(pattern: str) -> List[Path]:
    root = Path.cwd()
    matches = list(root.glob(pattern))
    # Also check nested under results/ and logs/
    matches += list((root / "results").glob(pattern))
    matches += list((root / "logs").glob(pattern))
    # Deduplicate
    uniq: List[Path] = []
    seen = set()
    for p in matches:
        try:
            key = p.resolve()
        except Exception:
            key = p
        if key not in seen and p.exists():
            uniq.append(p)
            seen.add(key)
    return uniq


def move_to_backup(paths: List[Path], backup_dir: Path, apply: bool) -> None:
    backup_dir.mkdir(parents=True, exist_ok=True)
    for p in paths:
        dest = backup_dir / p.name
        print(f"- move {p} -> {dest}")
        if apply:
            try:
                shutil.move(str(p), str(dest))
            except Exception as e:
                print(f"  warn: could not move {p}: {e}")


def prune_backups(root_backup: Path, keep: int, apply: bool) -> None:
    if not root_backup.exists():
        return
    backups = sorted((d for d in root_backup.iterdir() if d.is_dir()), key=lambda d: d.stat().st_mtime, reverse=True)
    if len(backups) > keep:
        to_delete = backups[keep:]
        print(f"Pruning old backups (keep={keep}):")
        for d in to_delete:
            print(f"- rm -rf {d}")
            if apply:
                shutil.rmtree(d, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description="Safe housekeeping: move artifacts to backup/<timestamp>")
    parser.add_argument("--apply", action="store_true", help="perform actions (default is dry-run)")
    parser.add_argument("--keep", type=int, default=10, help="number of most recent backups to keep")
    args = parser.parse_args()

    apply = args.apply
    keep = max(1, args.keep)

    root = Path.cwd()
    backup_root = root / "backup"
    backup_dir = backup_root / ts()

    print(f"Housekeeping (dry-run={not apply})")

    # Collect paths
    to_move: List[Path] = []
    for pattern in ARTIFACT_PATTERNS:
        paths = resolve_paths(pattern)
        to_move.extend(paths)

    # Deduplicate while preserving order
    seen = set()
    unique_to_move: List[Path] = []
    for p in to_move:
        if p.resolve() not in seen:
            unique_to_move.append(p)
            seen.add(p.resolve())

    if not unique_to_move:
        print("No artifacts found to move.")
    else:
        move_to_backup(unique_to_move, backup_dir, apply)

    prune_backups(backup_root, keep, apply)

    print(f"Done. Backup directory: {backup_dir}")


if __name__ == "__main__":
    main()

