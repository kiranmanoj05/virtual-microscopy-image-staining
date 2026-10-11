"""Prepare a fixed middle-plane pilot with scaling fitted on training fields only."""

import hashlib
import json
from pathlib import Path

import numpy as np

from .paired_planes import load_pair


def fit_scaling(pairs: list[tuple[np.ndarray, np.ndarray]]) -> dict:
    if not pairs:
        raise ValueError("training pairs are required to fit scaling")
    result = {}
    for index, name in enumerate(("input", "target")):
        values = np.concatenate([pair[index].reshape(-1) for pair in pairs]).astype(np.float32)
        if not np.isfinite(values).all():
            raise ValueError("non-finite training values")
        low, high = np.percentile(values, [1, 99])
        if high <= low:
            raise ValueError(f"training {name} has no percentile range")
        result[name] = {"low": float(low), "high": float(high),
                        "clip": name == "input"}
    result["fitted_on"] = "train only"
    return result


def scale_plane(array: np.ndarray, params: dict) -> np.ndarray:
    scaled = (array.astype(np.float32) - params["low"]) / (params["high"] - params["low"])
    return np.clip(scaled, 0, 1) if params["clip"] else scaled


def prepare_training_data(audit_path: Path, output_dir: Path) -> dict:
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if not audit.get("valid") or audit.get("failed_sources") or audit.get("test_pixels_loaded"):
        raise ValueError("need a successful audit with test pixels unopened")
    if output_dir.exists():
        raise ValueError("training data output already exists")
    raw_pairs = []
    training_pairs = []
    seen_sources, group_splits = set(), {}
    for field in audit["fields"]:
        if field["source_id"] in seen_sources:
            raise ValueError("duplicate source in audited fields")
        seen_sources.add(field["source_id"])
        previous = group_splits.setdefault(field["group_id"], field["split"])
        if previous != field["split"]:
            raise ValueError("acquisition group crosses splits")
        if field["split"] == "test":
            continue
        if field["split"] not in ("train", "validation") or field.get("errors"):
            raise ValueError("invalid development field in audit")
        path = Path(field["path"])
        # Bind prepared data to the actual audited bytes, including after an intervening edit.
        receipt_path = path.with_name(path.name.replace(".ome.tiff", ".json"))
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        if (digest.hexdigest() != receipt["sha256"] or path.stat().st_size != receipt["size_bytes"] or
                receipt["pilot_plan_sha256"] != audit["pilot_plan_sha256"] or
                any(receipt[key] != field[key] for key in ("source_id", "group_id", "split"))):
            raise ValueError("source changed since audit or receipt identities disagree")
        series = field["tiff"]["series"][0]
        dimensions = dict(zip(series["axes"], series["shape"]))
        z = dimensions.get("Z", 1) // 2
        pair = load_pair(path, z=z, input_channel=field["input_channel_metadata"]["index"],
                         target_channel=field["target_channel_metadata"]["index"])
        raw_pairs.append((field, pair))
        if field["split"] == "train":
            training_pairs.append((pair.input, pair.target))
    scaling = fit_scaling(training_pairs)
    output_dir.mkdir(parents=True)
    entries = []
    for index, (field, pair) in enumerate(raw_pairs):
        input_file, target_file = f"{index:03d}_input.npy", f"{index:03d}_target.npy"
        np.save(output_dir / input_file, pair.input, allow_pickle=False)
        np.save(output_dir / target_file, pair.target, allow_pickle=False)
        entries.append({"source_id": field["source_id"], "group_id": field["group_id"],
                        "split": field["split"], "z": pair.metadata["z"],
                        "input_file": input_file, "target_file": target_file,
                        "shape": list(pair.input.shape), "raw_dtype": str(pair.input.dtype),
                        "input_sha256": hashlib.sha256((output_dir / input_file).read_bytes()).hexdigest(),
                        "target_sha256": hashlib.sha256((output_dir / target_file).read_bytes()).hexdigest()})
    result = {"pilot_plan_sha256": audit["pilot_plan_sha256"],
              "depth_policy": "floor(SizeZ/2), fixed for every field",
              "scaling": scaling, "fields": entries, "test_pixels_loaded": False,
              "audit_sha256": hashlib.sha256(audit_path.read_bytes()).hexdigest(),
              "registration_status": audit["registration_status"]}
    canonical = json.dumps(result, sort_keys=True, separators=(",", ":"))
    result["prepared_metadata_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    (output_dir / "dataset.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return {key: value for key, value in result.items() if key != "fields"} | {
        "train_fields": sum(f["split"] == "train" for f in entries),
        "validation_fields": sum(f["split"] == "validation" for f in entries),
        "output_dir": str(output_dir)}
