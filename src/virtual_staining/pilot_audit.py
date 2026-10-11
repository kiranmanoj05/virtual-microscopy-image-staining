"""Audit local pilot bytes, channel metadata and development-plane decoding."""

import hashlib
import json
from collections import Counter
from pathlib import Path

from .image_info import read_tiff_info


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_pilot(plan_path: Path, data_dir: Path, output_dir: Path) -> dict:
    from .paired_planes import load_pair
    from .preview import preview_pair

    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    fingerprint = plan.pop("plan_sha256")
    canonical = json.dumps(plan, sort_keys=True, separators=(",", ":"))
    if hashlib.sha256(canonical.encode()).hexdigest() != fingerprint:
        raise ValueError("pilot plan fingerprint mismatch")
    if output_dir.exists():
        raise ValueError("audit output already exists")
    fields = []
    preview_field = None
    for field in plan["fields"]:
        fov = field["fov_id"]
        if not fov.isdigit():
            raise ValueError("expected a numeric FOV identity")
        path = data_dir / f"allen_fov_{fov}.ome.tiff"
        receipt_path = data_dir / f"allen_fov_{fov}.json"
        result = {"source_id": field["source_id"], "group_id": field["group_id"],
                  "split": field["split"], "path": str(path), "errors": [],
                  "decoded_planes": [], "pixels_loaded": False}
        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            for key, expected in {
                "source_id": field["source_id"], "group_id": field["group_id"],
                "split": field["split"], "download_url": field["download_url"],
                "size_bytes": field["size_bytes"], "pilot_plan_sha256": fingerprint,
            }.items():
                if receipt.get(key) != expected:
                    raise ValueError(f"receipt mismatch: {key}")
            if path.stat().st_size != field["size_bytes"] or _hash(path) != receipt.get("sha256"):
                raise ValueError("local file failed size/checksum verification")
            info = read_tiff_info(path)
            result["tiff"] = info
            if len(info["series"]) != 1 or len(info["ome_images"]) != 1:
                raise ValueError("expected a single-series OME field")
            series = info["series"][0]
            axes, shape = series["axes"], series["shape"]
            if (not axes.endswith("YX") or "C" not in axes[:-2] or
                    len(set(axes)) != len(axes) or any(a not in "TCZ" for a in axes[:-2])):
                raise ValueError("unsupported scalar-plane axes")
            dimensions = dict(zip(axes, shape))
            if dimensions.get("T", 1) != 1:
                raise ValueError("multiple timepoints require explicit selection")
            channels = info["ome_images"][0]["channels"]
            if len(channels) != dimensions["C"]:
                raise ValueError("OME channel count does not match series")
            input_index, target_index = field["input_channel"], field["target_channel"]
            if input_index == target_index or not all(0 <= c < len(channels) for c in (input_index, target_index)):
                raise ValueError("invalid paired channel indices")
            input_meta, target_meta = channels[input_index], channels[target_index]
            if "brightfield" not in input_meta.get("Fluor", "").lower():
                raise ValueError("input channel is not identified as brightfield")
            if "hoechst" not in target_meta.get("Fluor", "").lower():
                raise ValueError("target channel is not identified as Hoechst DNA fluorescence")
            result["input_channel_metadata"] = input_meta
            result["target_channel_metadata"] = target_meta
            depths = sorted({int(dimensions.get("Z", 1) * fraction) for fraction in (0.25, 0.5, 0.75)})
            result["qc_z_indices"] = depths
            # Test pixels remain unopened during development; metadata and bytes are still audited.
            if field["split"] != "test":
                for z in depths:
                    pair = load_pair(path, z=z, input_channel=input_index, target_channel=target_index)
                    stats = {name: {"min": float(array.min()), "max": float(array.max()),
                                    "mean": float(array.mean()), "std": float(array.std())}
                             for name, array in (("input", pair.input), ("target", pair.target))}
                    result["decoded_planes"].append({"z": z, "metadata": pair.metadata, "intensities": stats})
                result["pixels_loaded"] = True
                if field["split"] == "train" and preview_field is None:
                    preview_field = (field, path, depths)
        except (OSError, ValueError, KeyError, IndexError) as error:
            result["errors"].append(str(error))
        fields.append(result)
    summary = {
        "valid": all(not field["errors"] for field in fields) and bool(fields),
        "pilot_plan_sha256": fingerprint, "field_count": len(fields),
        "source_counts": dict(Counter(field["split"] for field in fields)),
        "failed_sources": [field["source_id"] for field in fields if field["errors"]],
        "test_pixels_loaded": any(field["pixels_loaded"] for field in fields if field["split"] == "test"),
        "registration_status": "unverified; preview review is required",
        "model_trained": False, "fields": fields,
    }
    output_dir.mkdir(parents=True)
    if summary["valid"] and preview_field:
        field, path, depths = preview_field
        summary["training_preview"] = preview_pair(
            path, z_indices=depths, input_channel=field["input_channel"],
            target_channel=field["target_channel"], out_dir=output_dir / "training-preview")
        summary["preview_source_id"] = field["source_id"]
    (output_dir / "audit.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return {key: value for key, value in summary.items() if key != "fields"} | {"output_dir": str(output_dir)}
