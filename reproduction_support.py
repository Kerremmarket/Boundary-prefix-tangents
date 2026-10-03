"""Integrity checks for this public source export and locally generated inputs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hashes(root: Path) -> dict[str, str]:
    return dict(line.split("  ", 1)[::-1] for line in
                (root / "SOURCE_MANIFEST.sha256").read_text().splitlines())


def validate_source_manifest(root: Path) -> None:
    for relative, expected in source_hashes(root).items():
        path = root / relative
        if not path.is_file() or digest(path) != expected:
            raise RuntimeError(f"Public source manifest mismatch: {relative}")


def register_local_files(root: Path, paths, *, stage: str) -> None:
    """Record generated files; this is provenance, not a match-to-paper claim."""
    record_path = root / "local-reproduction.json"
    record = json.loads(record_path.read_text()) if record_path.exists() else {"files": {}}
    for path in paths:
        path = Path(path).resolve()
        relative = path.relative_to(root.resolve()).as_posix()
        record["files"][relative] = {"sha256": digest(path), "stage": stage}
    record["meaning"] = "Locally generated inputs/results; not certification of historical equality."
    record_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")


def validate_reproduction_file(path: Path, historical_hash: str, root: Path) -> str:
    path, root = path.resolve(), root.resolve()
    relative = path.relative_to(root).as_posix()
    if not path.is_file():
        raise RuntimeError(f"Missing {relative}; follow DATA_ACCESS.md and reproduce.py stages.")
    observed = digest(path)
    sources = source_hashes(root)
    if relative in sources:
        expected = sources[relative]
    else:
        record_path = root / "local-reproduction.json"
        record = json.loads(record_path.read_text()) if record_path.exists() else {"files": {}}
        entry = record["files"].get(relative)
        expected = entry["sha256"] if entry else historical_hash
    if observed != expected:
        raise RuntimeError(f"Input changed since preparation: {relative}")
    return observed
