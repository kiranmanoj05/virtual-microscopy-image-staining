"""Check identities and file integrity before an image loader touches the data.

This audit cannot establish biological independence or image registration.
Those checks require the acquisition metadata and actual image inspection.
"""

import csv
import hashlib
from collections import Counter
from pathlib import Path

FIELDS = (
    "pair_id", "input_path", "target_path", "source_id", "group_id", "structure",
    "split", "input_sha256", "target_sha256",
)
SPLITS = {"train", "validation", "test"}


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_manifest(manifest: Path, data_root: Path) -> dict:
    root = data_root.resolve()
    errors = []
    counts = Counter()
    pair_ids = set()
    assignments = {"source_id": {}, "group_id": {}}
    content_splits = {}
    hash_cache = {}
    row_count = 0
    with manifest.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        missing = set(FIELDS) - set(reader.fieldnames or [])
        if missing:
            return {"valid": False, "rows": 0, "counts": {},
                    "errors": ["missing columns: " + ", ".join(sorted(missing))]}
        for line, row in enumerate(reader, start=2):
            row_count += 1
            prefix = f"line {line}: "
            if None in row or any(not (row.get(field) or "").strip() for field in FIELDS):
                errors.append(prefix + "malformed row or empty required field")
                continue
            row = {key: value.strip() for key, value in row.items() if value is not None}
            pair_id, split = row["pair_id"], row["split"]
            if pair_id in pair_ids:
                errors.append(prefix + f"duplicate pair_id {pair_id}")
            pair_ids.add(pair_id)
            if split not in SPLITS:
                errors.append(prefix + f"unknown split {split}")
                continue
            counts[split] += 1
            # Keep original fields and acquisition groups together, even if
            # the same field is listed again for another target channel.
            for key, mapping in assignments.items():
                identity = row[key]
                previous = mapping.setdefault(identity, split)
                if previous != split:
                    errors.append(prefix + f"{key} {identity} crosses splits")
            for role in ("input", "target"):
                relative = Path(row[f"{role}_path"])
                path = (root / relative).resolve()
                if relative.is_absolute() or not path.is_relative_to(root):
                    errors.append(prefix + f"{role} path must stay inside data root")
                    continue
                if not path.is_file():
                    errors.append(prefix + f"missing {role} file: {relative}")
                    continue
                if path not in hash_cache:
                    hash_cache[path] = file_hash(path)
                actual = hash_cache[path]
                if row[f"{role}_sha256"].lower() != actual:
                    errors.append(prefix + f"{role} checksum mismatch")
                # Compare bytes across both roles: renaming a file or changing
                # its group ID must not let a duplicate evade the audit.
                previous = content_splits.setdefault(actual, split)
                if previous != split:
                    errors.append(prefix + f"duplicate file content crosses splits ({role})")
    if not row_count:
        errors.append("manifest contains no observations")
    return {"valid": not errors, "rows": row_count,
            "counts": dict(sorted(counts.items())), "errors": errors}
