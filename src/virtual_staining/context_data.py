"""Prepare a provenance-bound neighbouring-input comparison, development only."""

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

from .image_info import read_tiff_info
from .paired_planes import load_context
from .pilot import _url
from .pilot_audit import _hash
from .trainer import load_development_data
from .training_data import scale_plane
from urllib.parse import urlencode


def _read_frozen(path, key):
    document = json.loads(path.read_text())
    fingerprint = document.pop(key)
    canonical = json.dumps(document, sort_keys=True, separators=(",", ":"))
    if hashlib.sha256(canonical.encode()).hexdigest() != fingerprint:
        raise ValueError(f"{key} fingerprint mismatch")
    return document, fingerprint


def prepare_context_data(plan_path: Path, baseline_path: Path, receipts_path: Path,
                         data_dir: Path, output_dir: Path) -> dict:
    """Bind z-1/z/z+1 inputs to the existing middle-plane baseline.

    Scaling is copied, never refitted. Test paths are never opened. Raw file
    hashes must match the original receipts, not just newly observed bytes.
    Every development field is required: no silent filtering or split reroll.
    """
    if output_dir.exists():
        raise ValueError("context output already exists")
    plan, plan_sha = _read_frozen(plan_path, "plan_sha256")
    baseline, baseline_sha, raw_baseline = load_development_data(baseline_path, scaled=False)
    if (baseline["pilot_plan_sha256"] != plan_sha or
            baseline["depth_policy"] != "floor(SizeZ/2), fixed for every field"):
        raise ValueError("baseline plan or middle-depth policy mismatch")
    receipts = json.loads(receipts_path.read_text())
    if receipts["pilot_plan_sha256"] != plan_sha:
        raise ValueError("receipt report belongs to another plan")
    by_receipt = {r["source_id"]: r for r in receipts["records"]}
    by_plan, group_splits = {}, {}
    for field in plan["fields"]:
        if field["source_id"] in by_plan:
            raise ValueError("duplicate plan source")
        by_plan[field["source_id"]] = field
        if field["split"] not in ("train", "validation", "test"):
            raise ValueError("invalid plan split")
        previous = group_splits.setdefault(field["group_id"], field["split"])
        if previous != field["split"]:
            raise ValueError("plan group crosses splits")
    if len(by_receipt) != len(receipts["records"]):
        raise ValueError("duplicate receipt source")
    development = {s for s, f in by_plan.items() if f["split"] != "test"}
    if development != {f[0]["source_id"] for f in raw_baseline}:
        raise ValueError("baseline development source identities differ from plan")
    prepared = []
    for baseline_field, middle, target in raw_baseline:
        source = baseline_field["source_id"]
        field, receipt = by_plan[source], by_receipt[source]
        if any(baseline_field[k] != field[k] for k in ("source_id", "group_id", "split")):
            raise ValueError("baseline source group or split differs from plan")
        fov = field["fov_id"]
        if not fov.isdigit() or not field["version_id"]:
            raise ValueError("numeric FOV and pinned object version are required")
        expected_url = _url(field["source_path"]) + "?" + urlencode({"versionId": field["version_id"]})
        contract = {k: field[k] for k in ("source_id", "group_id", "split", "size_bytes")}
        contract |= {"download_url": expected_url, "pilot_plan_sha256": plan_sha}
        if field["download_url"] != expected_url or any(receipt.get(k) != v for k, v in contract.items()):
            raise ValueError("original receipt or versioned source contract mismatch")
        path = data_dir / f"allen_fov_{fov}.ome.tiff"
        if path.stat().st_size != field["size_bytes"] or _hash(path) != receipt["sha256"]:
            raise ValueError("raw source checksum mismatch against original receipt")
        info = read_tiff_info(path)
        if len(info["series"]) != 1 or len(info["ome_images"]) != 1:
            raise ValueError("expected a single-series OME field")
        series = info["series"][0]
        dimensions = dict(zip(series["axes"], series["shape"]))
        channels = info["ome_images"][0]["channels"]
        ic, tc = field["input_channel"], field["target_channel"]
        if (len(channels) != dimensions.get("C") or ic == tc or
                not all(0 <= c < len(channels) for c in (ic, tc)) or
                "brightfield" not in channels[ic].get("Fluor", "").lower() or
                "hoechst" not in channels[tc].get("Fluor", "").lower()):
            raise ValueError("unverified brightfield/DNA channel metadata")
        z = dimensions.get("Z", 1) // 2
        if baseline_field["z"] != z:
            raise ValueError("baseline middle depth changed")
        pair = load_context(path, z=z, input_channel=ic, target_channel=tc)
        if (pair.input[1].dtype != middle.dtype or pair.target.dtype != target.dtype or
                not np.array_equal(pair.input[1], middle) or not np.array_equal(pair.target, target)):
            raise ValueError("centre input or target differs from frozen baseline pixels")
        prepared.append((baseline_field, pair, {
            "raw_sha256": receipt["sha256"], "raw_size_bytes": field["size_bytes"],
            "source_path": field["source_path"], "version_id": field["version_id"],
            "input_channel_metadata": channels[ic], "target_channel_metadata": channels[tc]}))
    # All identity/channel/boundary/pixel checks precede output creation.
    output_dir.mkdir(parents=True)
    entries = []
    for index, (field, pair, provenance) in enumerate(prepared):
        context_name = f"{index:03d}_context.npy"
        np.save(output_dir / context_name, pair.input, allow_pickle=False)
        middle_name, target_name = f"{index:03d}_middle.npy", f"{index:03d}_target.npy"
        for key, name in (("input_file", middle_name), ("target_file", target_name)):
            shutil.copyfile(baseline_path.parent / field[key], output_dir / name)
        entries.append({k: field[k] for k in ("source_id", "group_id", "split", "z")} | {
            "context_file": context_name, "middle_input_file": middle_name, "target_file": target_name,
            "context_sha256": _hash(output_dir / context_name),
            "middle_input_sha256": field["input_sha256"], "target_sha256": field["target_sha256"],
            "shape": list(pair.input.shape), "raw_dtype": str(pair.input.dtype),
            "planes": pair.metadata, **provenance})
    result = {"schema": "neighbouring-input-development-v1", "fields": entries,
              "pilot_plan_sha256": plan_sha, "baseline_prepared_sha256": baseline_sha,
              "original_receipts_sha256": _hash(receipts_path),
              "scaling": baseline["scaling"], "input_normalisation": "global",
              "depth_policy": baseline["depth_policy"], "input_offsets": [-1, 0, 1],
              "input_axes": "CYX", "target_axes": "YX", "target_offset": 0,
              "test_pixels_loaded": False, "registration_status": baseline["registration_status"],
              "limitations": ["Same-array coordinates do not prove biological co-registration.",
                              "Requires an explicit context policy, not the default single-input trainer."]}
    canonical = json.dumps(result, sort_keys=True, separators=(",", ":"))
    result["context_metadata_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    (output_dir / "context_dataset.json").write_text(json.dumps(result, indent=2) + "\n")
    return {k: v for k, v in result.items() if k != "fields"} | {
        "field_count": len(entries), "train_fields": sum(f["split"] == "train" for f in entries),
        "validation_fields": sum(f["split"] == "validation" for f in entries), "output_dir": str(output_dir)}


def load_context_data(path: Path, *, scaled=True):
    """Verify raw context and centre controls; optionally apply the frozen global scale."""
    metadata, identity = _read_frozen(path, "context_metadata_sha256")
    if (metadata["schema"] != "neighbouring-input-development-v1" or metadata["test_pixels_loaded"] or
            metadata["input_offsets"] != [-1, 0, 1] or metadata["target_offset"] != 0 or
            metadata["input_axes"] != "CYX" or metadata["target_axes"] != "YX" or
            metadata["input_normalisation"] != "global" or metadata["scaling"]["fitted_on"] != "train only"):
        raise ValueError("invalid context preparation contract")
    data, seen, groups = [], set(), {}
    for field in metadata["fields"]:
        if field["split"] not in ("train", "validation") or field["source_id"] in seen:
            raise ValueError("expected distinct development sources")
        seen.add(field["source_id"])
        if groups.setdefault(field["group_id"], field["split"]) != field["split"]:
            raise ValueError("context group crosses splits")
        arrays = []
        for key in ("context", "middle_input", "target"):
            name = field[key + "_file"]
            if Path(name).name != name or _hash(path.parent / name) != field[key + "_sha256"]:
                raise ValueError("context pixels failed basename/checksum verification")
            array = np.load(path.parent / name, allow_pickle=False)
            if not np.isfinite(array).all():
                raise ValueError("non-finite context values")
            arrays.append(array)
        context, middle, target = arrays
        if (context.ndim != 3 or context.shape[0] != 3 or middle.ndim != 2 or
                min(middle.shape) < 8 or context.shape[1:] != middle.shape or target.shape != middle.shape or
                context.dtype != middle.dtype or not np.array_equal(context[1], middle)):
            raise ValueError("context shape or centre-input control mismatch")
        if scaled:
            context = scale_plane(context, metadata["scaling"]["input"])
            target = scale_plane(target, metadata["scaling"]["target"])
        data.append((field, context, target))
    if not all(any(f[0]["split"] == split for f in data) for split in ("train", "validation")):
        raise ValueError("both development splits are required")
    return metadata, identity, data
