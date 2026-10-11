"""Collapse Allen cell-level metadata into a conservative field inventory."""

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path, PurePosixPath


REQUIRED = ("FOVId", "PlateId", "WellId", "CellLine", "SourceReadPath",
            "ChannelNumberBrightfield", "ChannelNumber405")
OPTIONAL = ("RunId", "InstrumentId", "Structure", "Gene")


def read_allen_inventory(path: Path, *, cell_line: str) -> dict:
    """Preserve observed identities; plate grouping is a provisional boundary."""
    if not cell_line.strip():
        raise ValueError("cell_line must be nonempty")
    fields = {}
    paths = {}
    rows_read = 0
    matched_rows = 0
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        missing = set(REQUIRED) - set(reader.fieldnames or [])
        if missing:
            raise ValueError("missing columns: " + ", ".join(sorted(missing)))
        for line, row in enumerate(reader, start=2):
            rows_read += 1
            if (row.get("CellLine") or "").strip() != cell_line:
                continue
            matched_rows += 1
            values = {key: (row.get(key) or "").strip() for key in REQUIRED + OPTIONAL}
            if any(not values[key] for key in REQUIRED):
                raise ValueError(f"line {line}: missing required field metadata")
            relative = PurePosixPath(values["SourceReadPath"])
            if (relative.is_absolute() or ".." in relative.parts or
                    not relative.parts or relative.parts[0] != "fovs" or
                    "\\" in values["SourceReadPath"]):
                raise ValueError(f"line {line}: expected a safe fovs/ source path")
            channels = []
            for key in ("ChannelNumberBrightfield", "ChannelNumber405"):
                try:
                    channel = int(values[key])
                except ValueError as error:
                    raise ValueError(f"line {line}: invalid channel {key}") from error
                if channel < 0:
                    raise ValueError(f"line {line}: negative channel {key}")
                channels.append(channel)
            if channels[0] == channels[1]:
                raise ValueError(f"line {line}: input and target channels must differ")
            source_id = "allen-fov:" + values["FOVId"]
            item = {
                "source_id": source_id,
                "group_id": "allen-plate:" + values["PlateId"],
                "fov_id": values["FOVId"], "plate_id": values["PlateId"],
                "well_id": values["WellId"], "cell_line": values["CellLine"],
                "source_path": values["SourceReadPath"],
                "input_channel": channels[0], "target_channel": channels[1],
                **{key: values[key] for key in OPTIONAL},
            }
            previous = fields.get(source_id)
            if previous is not None and previous != item:
                raise ValueError(f"line {line}: conflicting metadata for {source_id}")
            owner = paths.setdefault(item["source_path"], source_id)
            if owner != source_id:
                raise ValueError(f"line {line}: source path shared by distinct FOV IDs")
            fields[source_id] = item
    if not fields:
        raise ValueError(f"no fields found for cell line {cell_line}")
    ordered = sorted(fields.values(), key=lambda item: (item["group_id"], item["source_id"]))
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {
        "metadata_sha256": digest.hexdigest(), "cell_line": cell_line,
        "rows_read": rows_read, "matched_cell_rows": matched_rows,
        "field_count": len(ordered),
        "plate_count": len({item["plate_id"] for item in ordered}),
        "well_count": len({(item["plate_id"], item["well_id"]) for item in ordered}),
        "fields_per_plate": dict(sorted(Counter(item["group_id"] for item in ordered).items())),
        "grouping": "PlateId (provisional; acquisition dependencies require review)",
        "warnings": [
            "Metadata grouping does not establish independence or pixel registration.",
            "Source paths are unversioned metadata references, not verified download URLs.",
            "No split has been selected and no image pixels have been inspected by this command.",
        ],
        "fields": ordered,
    }


def write_allen_inventory(report: dict, output_dir: Path) -> dict:
    """Write metadata and split-compatible identities without overwriting a review."""
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "inventory.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with (output_dir / "sources.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("source_id", "group_id"))
        writer.writeheader()
        writer.writerows({key: item[key] for key in writer.fieldnames} for item in report["fields"])
    return {key: value for key, value in report.items() if key != "fields"} | {"output_dir": str(output_dir)}
