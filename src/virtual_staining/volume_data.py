"""Version-bound development volumes for a reference-style 3D baseline."""

import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
from scipy.ndimage import zoom

from .context_data import _read_frozen
from .image_info import read_tiff_info
from .paired_planes import _load_selected
from .pilot import _url
from .pilot_audit import _hash
from urllib.parse import urlencode


def standardise_volume(raw):
    """Per-volume z-score; target stats define relative, not raw, fluorescence."""
    a = np.asarray(raw, dtype=np.float64)
    if a.ndim != 3 or not np.isfinite(a).all():
        raise ValueError("expected finite ZYX volume")
    mean, std = float(a.mean()), float(a.std())
    if std <= 0:
        raise ValueError("constant volume cannot be standardised")
    return (a - mean) / std, {"mean": mean, "std": std}


def prepare_volumes(plan_path, receipts_path, raw_dir, output_dir, *, voxel_um=0.29):
    """Decode only brightfield/DNA in development sources, never old test files."""
    if output_dir.exists():
        raise ValueError("volume output already exists")
    if not np.isfinite(voxel_um) or voxel_um <= 0:
        raise ValueError("positive finite voxel spacing required")
    plan, plan_sha = _read_frozen(plan_path, "plan_sha256")
    receipts = json.loads(receipts_path.read_text())
    if receipts["pilot_plan_sha256"] != plan_sha:
        raise ValueError("original receipts belong to another plan")
    by_receipt = {f["source_id"]: f for f in receipts["records"]}
    if len(by_receipt) != len(receipts["records"]):
        raise ValueError("duplicate receipt source")
    sources, groups = set(), {}
    for f in plan["fields"]:
        if f["source_id"] in sources or f["split"] not in ("train", "validation", "test"):
            raise ValueError("duplicate source or invalid split")
        sources.add(f["source_id"])
        if groups.setdefault(f["group_id"], f["split"]) != f["split"]:
            raise ValueError("group crosses splits")
    development = [f for f in plan["fields"] if f["split"] != "test"]
    if not all(any(f["split"] == s for f in development) for s in ("train", "validation")):
        raise ValueError("both development splits required")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".volumes-", dir=output_dir.parent))
    try:
        entries = []
        for index, f in enumerate(development):
            receipt = by_receipt[f["source_id"]]
            expected_url = _url(f["source_path"]) + "?" + urlencode({"versionId": f["version_id"]})
            if (not f["fov_id"].isdigit() or not f["version_id"] or f["download_url"] != expected_url or
                    any(receipt[k] != f[k] for k in ("source_id", "group_id", "split", "size_bytes", "download_url"))):
                raise ValueError("source/receipt version contract mismatch")
            path = raw_dir / f"allen_fov_{f['fov_id']}.ome.tiff"
            if path.stat().st_size != f["size_bytes"] or _hash(path) != receipt["sha256"]:
                raise ValueError("raw source differs from original receipt")
            info = read_tiff_info(path)
            if len(info["ome_images"]) != 1 or len(info["series"]) != 1:
                raise ValueError("single-series OME data required")
            ome = info["ome_images"][0]
            spacing = []
            for axis in "ZYX":
                if ome["pixels"].get("PhysicalSize" + axis + "Unit", "µm") not in ("µm", "um"):
                    raise ValueError("OME physical spacing must be micrometres")
                value = float(ome["pixels"]["PhysicalSize" + axis])
                if not np.isfinite(value) or value <= 0:
                    raise ValueError("invalid OME physical spacing")
                spacing.append(value)
            dims = dict(zip(info["series"][0]["axes"], info["series"][0]["shape"]))
            ic, tc = f["input_channel"], f["target_channel"]
            channels = ome["channels"]
            if (ic == tc or not all(0 <= c < len(channels) for c in (ic, tc)) or
                    "brightfield" not in channels[ic].get("Fluor", "").lower() or
                    "hoechst" not in channels[tc].get("Fluor", "").lower()):
                raise ValueError("unverified brightfield/DNA channels")
            nz = dims["Z"]
            arrays, loaded = _load_selected(path, [(z, c) for c in (ic, tc) for z in range(nz)], 0)
            field = {k: f[k] for k in ("source_id", "group_id", "split", "version_id")}
            field |= {"raw_sha256": receipt["sha256"], "source_spacing_zyx_um": spacing,
                      "source_shape": info["series"][0]["shape"], "page_indices": loaded["page_indices"]}
            for name, pages in (("input", arrays[:nz]), ("target", arrays[nz:])):
                normalised, stats = standardise_volume(np.stack(pages))
                # Match release_1 train_model.py: normalise, then cubic resample.
                resampled = zoom(normalised, np.asarray(spacing) / voxel_um,
                                 order=3, mode="nearest").astype(np.float32)
                if min(resampled.shape) < 16 or not np.isfinite(resampled).all():
                    raise ValueError("prepared volume too small or non-finite")
                filename = f"{index:03d}_{name}.npy"
                np.save(staging / filename, resampled, allow_pickle=False)
                field[name + "_file"] = filename
                field[name + "_sha256"] = _hash(staging / filename)
                field[name + "_statistics"] = stats
                field[name + "_shape"] = list(resampled.shape)
            if field["input_shape"] != field["target_shape"]:
                raise ValueError("resampled pair shapes disagree")
            entries.append(field)
        report = {"schema": "reference-volumes-v1", "fields": entries,
                  "pilot_plan_sha256": plan_sha, "original_receipts_sha256": _hash(receipts_path),
                  "voxel_zyx_um": [voxel_um] * 3,
                  "preprocessing": "per-volume zscore then cubic resample; float32 output; no clipping",
                  "target_units": "relative per-volume fluorescence; no raw-intensity calibration",
                  "old_test_sources_excluded": True, "test_pixels_loaded": False,
                  "registration_status": "unverified; same-array coordinates are not registration evidence"}
        canonical = json.dumps(report, sort_keys=True, separators=(",", ":"))
        report["volume_metadata_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
        (staging / "volumes.json").write_text(json.dumps(report, indent=2) + "\n")
        staging.rename(output_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {k: v for k, v in report.items() if k != "fields"} | {"fields": len(entries)}


def load_volumes(path):
    metadata, identity = _read_frozen(path, "volume_metadata_sha256")
    if (metadata["schema"] != "reference-volumes-v1" or metadata["test_pixels_loaded"] or
            not metadata["old_test_sources_excluded"]):
        raise ValueError("development-only reference volumes required")
    fields, seen, groups = [], set(), {}
    for f in metadata["fields"]:
        if f["split"] not in ("train", "validation") or f["source_id"] in seen:
            raise ValueError("distinct development sources required")
        seen.add(f["source_id"])
        if groups.setdefault(f["group_id"], f["split"]) != f["split"]:
            raise ValueError("group crosses splits")
        arrays = []
        for key in ("input", "target"):
            name = f[key + "_file"]
            source = path.parent / name
            if Path(name).name != name or _hash(source) != f[key + "_sha256"]:
                raise ValueError("prepared volume checksum/basename mismatch")
            a = np.load(source, mmap_mode="r", allow_pickle=False)
            if (a.ndim != 3 or a.dtype != np.float32 or list(a.shape) != f[key + "_shape"] or
                    not np.isfinite(a).all()):
                raise ValueError("expected finite float32 ZYX volume matching metadata")
            arrays.append(a)
        if arrays[0].shape != arrays[1].shape:
            raise ValueError("paired volume shapes disagree")
        fields.append((f, *arrays))
    if not all(any(f[0]["split"] == s for f in fields) for s in ("train", "validation")):
        raise ValueError("both development splits required")
    return metadata, identity, fields
