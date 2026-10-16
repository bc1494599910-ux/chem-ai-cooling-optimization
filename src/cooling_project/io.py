"""Inspectable output and provenance without copying private source reports."""

import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Unsupported JSON value: {type(value).__name__}")


def dump_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False,
                                    default=json_default) + "\n", encoding="utf-8")


def new_output(path):
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise ValueError(f"Output directory is not empty: {path}. Choose a new --output directory.")
    path.mkdir(parents=True, exist_ok=True)
    return path


def dump_csv(path, rows):
    rows = list(rows)
    if not rows:
        Path(path).write_text("", encoding="utf-8")
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, keys)
        writer.writeheader()
        writer.writerows(rows)


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def manifest(config_path, output, started_at, total_seconds):
    sources = sorted(Path("src").rglob("*.py")) + sorted(Path("configs").glob("*.json"))
    sources += [p for p in (Path("pyproject.toml"), Path("uv.lock")) if p.exists()]
    hashes = {p.as_posix(): file_hash(p) for p in sources}
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    return {
        "started_at_utc": started_at, "total_wall_seconds": total_seconds,
        "python": sys.version.split()[0], "platform": platform.platform(),
        "processor": platform.processor(), "logical_cpus": os.cpu_count(),
        "packages": {p: importlib.metadata.version(p) for p in
                     ("numpy", "scipy", "scikit-learn", "matplotlib", "threadpoolctl")},
        "git_commit": commit, "working_tree_dirty": dirty,
        "config_file": Path(config_path).as_posix(), "config_sha256": file_hash(config_path),
        "source_sha256": hashes,
        "source_set_sha256": hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(),
        "output_sha256": {p.relative_to(output).as_posix(): file_hash(p) for p in sorted(output.rglob("*")) if p.is_file()},
        "command": f"python -m cooling_project --config {Path(config_path).as_posix()} experiment --output {Path(output).as_posix()}",
    }
