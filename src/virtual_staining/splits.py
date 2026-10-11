"""Create reproducible acquisition-group splits before patch extraction."""

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


INVENTORY_FIELDS = ("source_id", "group_id")
SPLIT_FIELDS = ("source_id", "group_id", "split")


def read_inventory(path: Path) -> list[dict[str, str]]:
    """Read one row per source and validate its declared acquisition group."""
    rows = []
    source_groups = {}
    seen_sources = set()
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        missing = set(INVENTORY_FIELDS) - set(reader.fieldnames or [])
        if missing:
            raise ValueError("missing columns: " + ", ".join(sorted(missing)))
        for line, row in enumerate(reader, start=2):
            source_id = (row.get("source_id") or "").strip()
            group_id = (row.get("group_id") or "").strip()
            if not source_id or not group_id:
                raise ValueError(f"line {line}: source_id and group_id are required")
            previous = source_groups.setdefault(source_id, group_id)
            if previous != group_id:
                raise ValueError(f"line {line}: source_id {source_id} belongs to multiple groups")
            if source_id in seen_sources:
                raise ValueError(f"line {line}: duplicate source_id {source_id}")
            seen_sources.add(source_id)
            rows.append({"source_id": source_id, "group_id": group_id})
    if not rows:
        raise ValueError("inventory contains no sources")
    return rows


def _group_rank(group_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}\0{group_id}".encode()).hexdigest()


def create_split_plan(rows: list[dict[str, str]], *, validation_groups: int,
                      test_groups: int, seed: int) -> dict:
    """Assign complete groups using a stable hash order, leaving at least one for train."""
    if validation_groups < 1 or test_groups < 1:
        raise ValueError("validation_groups and test_groups must both be at least one")
    group_ids = sorted({row["group_id"] for row in rows}, key=lambda value: (_group_rank(value, seed), value))
    held_out = validation_groups + test_groups
    if len(group_ids) <= held_out:
        raise ValueError(
            f"need more than {held_out} independent groups to leave at least one training group"
        )
    split_by_group = {}
    for index, group_id in enumerate(group_ids):
        if index < test_groups:
            split = "test"
        elif index < held_out:
            split = "validation"
        else:
            split = "train"
        split_by_group[group_id] = split
    assignments = [
        {"source_id": row["source_id"], "group_id": row["group_id"],
         "split": split_by_group[row["group_id"]]}
        for row in sorted(rows, key=lambda item: (item["group_id"], item["source_id"]))
    ]
    canonical = {
        "algorithm": "sha256-group-rank-v1",
        "seed": seed,
        "validation_groups": validation_groups,
        "test_groups": test_groups,
        "assignments": assignments,
    }
    fingerprint = hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    source_counts = Counter(row["split"] for row in assignments)
    group_counts = Counter(split_by_group.values())
    return {
        **canonical,
        "fingerprint_sha256": fingerprint,
        "source_counts": dict(sorted(source_counts.items())),
        "group_counts": dict(sorted(group_counts.items())),
    }


def write_split_plan(plan: dict, output_dir: Path) -> dict:
    """Write a frozen CSV plus metadata, refusing to replace an existing plan."""
    if output_dir.exists():
        raise FileExistsError(f"split output already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    csv_path = output_dir / "assignments.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=SPLIT_FIELDS)
        writer.writeheader()
        writer.writerows(plan["assignments"])
    metadata = {key: value for key, value in plan.items() if key != "assignments"}
    metadata["assignments_file"] = csv_path.name
    json_path = output_dir / "split.json"
    json_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return {**metadata, "output_dir": str(output_dir)}
