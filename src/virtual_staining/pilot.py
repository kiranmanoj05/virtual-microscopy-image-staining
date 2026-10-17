"""Small, reproducible Allen pilot selection and version-pinned downloads."""

import hashlib
import json
import shutil
import sys
import urllib.parse
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .splits import create_split_plan, write_split_plan

BASE_URL = "https://allencell.s3.amazonaws.com/aics/pipeline_integrated_cell/"


def select_pilot(inventory: dict, *, groups: int, fields_per_group: int, seed: int) -> dict:
    """Select by metadata only, with different wells within each plate."""
    if groups < 3 or fields_per_group < 1:
        raise ValueError("need at least three groups and one field per group")
    by_group = defaultdict(list)
    seen = set()
    for field in inventory["fields"]:
        if field["source_id"] in seen:
            raise ValueError("duplicate source identity in inventory")
        seen.add(field["source_id"])
        if field["group_id"] != "allen-plate:" + field["plate_id"]:
            raise ValueError("pilot requires plate-based inventory groups")
        by_group[field["group_id"]].append(field)
    rank = lambda value: hashlib.sha256(f"{seed}\0{value}".encode()).hexdigest()
    eligible = [group for group, fields in by_group.items()
                if len({f["well_id"] for f in fields}) >= fields_per_group]
    if len(eligible) < groups:
        raise ValueError("not enough plates with distinct wells for requested pilot")
    selected = []
    for group in sorted(eligible, key=lambda value: (rank(value), value))[:groups]:
        wells = set()
        for field in sorted(by_group[group], key=lambda f: (rank(f["source_id"]), f["source_id"])):
            if field["well_id"] in wells:
                continue
            selected.append(dict(field))
            wells.add(field["well_id"])
            if len(wells) == fields_per_group:
                break
    split = create_split_plan(selected, validation_groups=1, test_groups=1, seed=seed)
    assignments = {row["source_id"]: row["split"] for row in split["assignments"]}
    for field in selected:
        field["split"] = assignments[field["source_id"]]
    return {
        "selection_algorithm": "sha256-plate-field-rank-distinct-wells-v1",
        "metadata_sha256": inventory["metadata_sha256"],
        "cell_line": inventory["cell_line"], "seed": seed,
        "fields_per_group": fields_per_group, "split_plan": split,
        "fields": selected,
        "scope": "pipeline pilot; plate independence and registration remain unverified",
    }


def _url(source_path: str) -> str:
    parts = source_path.split("/")
    if (len(parts) != 2 or parts[0] != "fovs" or not parts[1] or
            "\\" in source_path or parts[1] in (".", "..")):
        raise ValueError("expected a safe fovs/ source path")
    return BASE_URL + urllib.parse.quote(source_path, safe="/")


def resolve_downloads(plan: dict) -> dict:
    """HEAD requests only: record immutable versions and exact sizes."""
    def resolve(field):
        url = _url(field["source_path"])
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=30) as response:
            version = response.headers.get("x-amz-version-id")
            size = int(response.headers.get("Content-Length", "0"))
        if not version or version == "null" or size <= 0:
            raise ValueError("download requires an S3 version ID and positive file size")
        return field | {"version_id": version, "size_bytes": size,
                        "download_url": url + "?" + urllib.parse.urlencode({"versionId": version})}
    with ThreadPoolExecutor(max_workers=4) as pool:
        plan["fields"] = list(pool.map(resolve, plan["fields"]))
    plan["total_size_bytes"] = sum(field["size_bytes"] for field in plan["fields"])
    canonical = json.dumps(plan, sort_keys=True, separators=(",", ":"))
    plan["plan_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    return plan


def write_pilot(plan: dict, output_dir: Path) -> dict:
    write_split_plan(plan["split_plan"], output_dir)
    (output_dir / "pilot.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    return {key: plan[key] for key in ("plan_sha256", "total_size_bytes", "scope")} | {
        "field_count": len(plan["fields"]),
        "source_counts": plan["split_plan"]["source_counts"],
        "group_counts": plan["split_plan"]["group_counts"], "output_dir": str(output_dir),
    }


def download_pilot(plan_path: Path, output_dir: Path, *, development_only=False) -> dict:
    """Download pinned TIFFs, recording observed hashes and refusing silent replacement."""
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    fingerprint = plan.pop("plan_sha256")
    canonical = json.dumps(plan, sort_keys=True, separators=(",", ":"))
    if hashlib.sha256(canonical.encode()).hexdigest() != fingerprint:
        raise ValueError("pilot plan fingerprint mismatch")
    if any(f["split"] not in ("train", "validation", "test") for f in plan["fields"]):
        raise ValueError("invalid split in pilot plan")
    fields = [f for f in plan["fields"] if not development_only or f["split"] != "test"]
    output_dir.mkdir(parents=True, exist_ok=True)
    remaining = sum(field["size_bytes"] for field in fields
                    if not (output_dir / f"allen_fov_{field['fov_id']}.ome.tiff").exists())
    if shutil.disk_usage(output_dir).free < remaining + 256 * 1024 * 1024:
        raise ValueError("insufficient free disk space for the recorded pilot size")
    records = []
    for index, field in enumerate(fields, start=1):
        expected_url = _url(field["source_path"]) + "?" + urllib.parse.urlencode({"versionId": field["version_id"]})
        if field["download_url"] != expected_url or not field["version_id"] or field["size_bytes"] <= 0:
            raise ValueError("invalid versioned download contract")
        fov = field["fov_id"]
        if not fov.isdigit():
            raise ValueError("expected a numeric FOV identity")
        path = output_dir / f"allen_fov_{fov}.ome.tiff"
        receipt_path = output_dir / f"allen_fov_{fov}.json"
        action = "Verifying" if path.exists() else "Downloading"
        print(f"{action} {index}/{len(fields)}: FOV {fov}", file=sys.stderr, flush=True)
        if path.exists():
            if not receipt_path.exists():
                raise ValueError(f"existing file has no receipt: {path}")
            receipt = json.loads(receipt_path.read_text())
            if receipt["download_url"] != expected_url:
                raise ValueError(f"existing file belongs to another version: {path}")
            if receipt["pilot_plan_sha256"] != fingerprint:
                raise ValueError(f"existing receipt belongs to another pilot plan: {path}")
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            if path.stat().st_size != field["size_bytes"] or digest.hexdigest() != receipt["sha256"]:
                raise ValueError(f"existing file failed integrity check: {path}")
            records.append(receipt)
            continue
        partial = path.with_suffix(path.suffix + ".part")
        digest = hashlib.sha256()
        count = 0
        try:
            with partial.open("xb") as stream:
                with urllib.request.urlopen(expected_url, timeout=60) as response:
                    if response.headers.get("x-amz-version-id") != field["version_id"]:
                        raise ValueError("server returned an unexpected object version")
                    while block := response.read(1024 * 1024):
                        count += len(block)
                        if count > field["size_bytes"]:
                            raise ValueError("download exceeds recorded file size")
                        digest.update(block)
                        stream.write(block)
            if count != field["size_bytes"]:
                raise ValueError("download is shorter than recorded file size")
        except Exception:
            # Keep the partial for diagnosis; a rerun must not overwrite it silently.
            raise
        partial.rename(path)
        receipt = {"source_id": field["source_id"], "split": field["split"],
                   "group_id": field["group_id"], "download_url": expected_url,
                   "size_bytes": count, "sha256": digest.hexdigest(), "path": str(path),
                   "pilot_plan_sha256": fingerprint}
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        records.append(receipt)
    return {"downloaded_or_verified_fields": len(records), "output_dir": str(output_dir),
            "pilot_plan_sha256": fingerprint, "development_only": development_only, "records": records}
