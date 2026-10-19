"""Evaluate a precommitted pair of models without making test-driven choices."""

import json
import importlib.metadata
import platform
import subprocess
import tempfile
from pathlib import Path
from urllib.parse import urlencode

import numpy as np
import torch
import tifffile

from .context_data import _read_frozen
from .image_info import read_tiff_info
from .model import UNet
from .paired_planes import load_context
from .pilot import _url
from .pilot_audit import _hash
from .trainer import image_scores, input_tensor, summarise_scores, _validation_preview
from .training_data import normalisation_fingerprint, scale_plane


METRICS = ["MAE", "Pearson (undefined for constant)", "SSIM data_range=1 window=7",
           "bright-target MAE at fixed scaled target >=1"]


def verify_evaluation(protocol_path, plan_path, root, expected_protocol_sha256):
    """Verify the externally retained lock and both checkpoints before pixel access.

    A self-consistent JSON hash is not evidence that choices preceded test access.
    The caller must supply the protocol fingerprint recorded before evaluation.
    """
    protocol, identity = _read_frozen(protocol_path, "protocol_sha256")
    plan, plan_sha = _read_frozen(plan_path, "plan_sha256")
    if identity != expected_protocol_sha256 or len(expected_protocol_sha256) != 64:
        raise ValueError("protocol differs from the precommitted fingerprint")
    if (protocol["protocol_version"] != 1 or protocol["test_pixels_loaded"] or
            protocol["status"] != "locked before test pixels opened" or
            protocol["pilot_plan_sha256"] != plan_sha or
            protocol["input_offsets"] != [-1, 0, 1] or protocol["target_offset"] != 0 or
            protocol["depth_policy"] != "floor(SizeZ/2), fixed for every field" or
            protocol["metrics"] != METRICS or protocol["scaling"]["fitted_on"] != "train only"):
        raise ValueError("unsupported frozen evaluation contract")
    for name in ("input", "target"):
        params = protocol["scaling"][name]
        if (not all(np.isfinite(params[k]) for k in ("low", "high")) or
                params["high"] <= params["low"] or params["clip"] != (name == "input")):
            raise ValueError("invalid frozen scaling")
    if not np.isfinite(protocol["constant_value"]):
        raise ValueError("invalid frozen constant")
    seen, groups, tests = set(), {}, []
    for field in plan["fields"]:
        if field["source_id"] in seen or field["split"] not in ("train", "validation", "test"):
            raise ValueError("duplicate source or invalid split")
        seen.add(field["source_id"])
        if groups.setdefault(field["group_id"], field["split"]) != field["split"]:
            raise ValueError("plan group crosses splits")
        if field["split"] == "test":
            tests.append(field)
    keys = ("source_id", "group_id", "fov_id", "version_id")
    if not tests or [{k: f[k] for k in keys} for f in tests] != protocol["test_fields"]:
        raise ValueError("test identities differ from frozen plan")
    policies = [entry["input_context"] for entry in protocol["models"]]
    if policies != ["repeated-centre", "neighbouring"]:
        raise ValueError("both matched fixed models are required")
    selected = min(protocol["models"], key=lambda m: m["validation_mae"])["input_context"]
    if protocol["validation_selected_context"] != selected:
        raise ValueError("validation-selected context mismatch")
    models, configs = {}, []
    for entry in protocol["models"]:
        path = root / entry["checkpoint_path"]
        if not path.resolve().is_relative_to(root.resolve()) or _hash(path) != entry["checkpoint_sha256"]:
            raise ValueError("frozen checkpoint path or checksum mismatch")
        saved = torch.load(path, map_location="cpu", weights_only=True)
        config = saved["config"]
        contract = {"input_context": entry["input_context"], "input_channels": 3,
                    "input_normalisation": "global", "pilot_plan_sha256": plan_sha,
                    "prepared_metadata_sha256": protocol["development_context_sha256"],
                    "depth_policy": protocol["depth_policy"], "git_revision": protocol["experiment_source"],
                    "preprocessing_sha256": normalisation_fingerprint(protocol, "global"),
                    "ssim_data_range_scaled": 1.0, "ssim_window": 7,
                    "parameter_count": entry["parameter_count"]}
        if (saved["step"] != entry["step"] or saved["scaling"] != protocol["scaling"] or
                any(config.get(k) != v for k, v in contract.items())):
            raise ValueError("checkpoint configuration differs from frozen protocol")
        model = UNet(config["width"], input_channels=3)
        model.load_state_dict(saved["model_state"])
        if sum(p.numel() for p in model.parameters()) != entry["parameter_count"]:
            raise ValueError("model capacity differs from protocol")
        models[entry["input_context"]] = model.eval()
        configs.append(config)
    if any(configs[0].get(k) != configs[1].get(k)
           for k in (configs[0].keys() | configs[1].keys()) - {"input_context"}):
        raise ValueError("model pair differs beyond context")
    return protocol, identity, tests, models


def evaluate_heldout(protocol_path: Path, plan_path: Path, receipts_path: Path,
                     checkpoint_root: Path, data_dir: Path, output_dir: Path, *,
                     expected_protocol_sha256: str):
    """CPU whole-plane inference on every locked field; output never overwrites.

    All source hashes and channel metadata are checked before any test decode.
    Outputs commit atomically; failed evaluation leaves no partial final report.
    """
    if output_dir.exists():
        raise ValueError("evaluation output already exists")
    protocol, identity, tests, models = verify_evaluation(
        protocol_path, plan_path, checkpoint_root, expected_protocol_sha256)
    receipts = json.loads(receipts_path.read_text())
    by_source = {r["source_id"]: r for r in receipts["records"]}
    if (len(by_source) != len(receipts["records"]) or
            receipts["pilot_plan_sha256"] != protocol["pilot_plan_sha256"]):
        raise ValueError("original receipt report identity mismatch")
    verified = []
    for field in tests:
        fov = field["fov_id"]
        if not fov.isdigit() or field["source_id"] != "allen-fov:" + fov or not field["version_id"]:
            raise ValueError("invalid frozen field identity")
        receipt = by_source[field["source_id"]]
        url = _url(field["source_path"]) + "?" + urlencode({"versionId": field["version_id"]})
        contract = {k: field[k] for k in ("source_id", "group_id", "split", "size_bytes")}
        contract |= {"download_url": url, "pilot_plan_sha256": protocol["pilot_plan_sha256"]}
        if field["download_url"] != url or any(receipt.get(k) != v for k, v in contract.items()):
            raise ValueError("original source receipt/version mismatch")
        path = data_dir / f"allen_fov_{fov}.ome.tiff"
        if path.stat().st_size != field["size_bytes"] or _hash(path) != receipt["sha256"]:
            raise ValueError("raw test source checksum mismatch")
        info = read_tiff_info(path)
        if len(info["series"]) != 1 or len(info["ome_images"]) != 1:
            raise ValueError("expected a single-series OME field")
        series = info["series"][0]
        dims = dict(zip(series["axes"], series["shape"]))
        axes = series["axes"]
        if (not axes.endswith("YX") or "C" not in axes[:-2] or
                len(set(axes)) != len(axes) or any(a not in "TCZ" for a in axes[:-2]) or
                dims.get("T", 1) != 1 or min(series["shape"][-2:]) < 8):
            raise ValueError("unsupported held-out scalar-plane axes")
        channels = info["ome_images"][0]["channels"]
        ic, tc = protocol["input_channel"], protocol["target_channel"]
        if (field["input_channel"] != ic or field["target_channel"] != tc or ic == tc or
                len(channels) != dims.get("C") or not all(0 <= c < len(channels) for c in (ic, tc)) or
                "brightfield" not in channels[ic].get("Fluor", "").lower() or
                "hoechst" not in channels[tc].get("Fluor", "").lower() or dims.get("Z", 1) < 3):
            raise ValueError("unverified brightfield/DNA channels or depth")
        z = dims["Z"] // 2
        with tifffile.TiffFile(path) as image:
            source = image.series[0]
            if len(source.pages) != int(np.prod(source.shape[:-2])):
                raise ValueError("expected one scalar YX page per selection")
            for depth, channel in [(z-1, ic), (z, ic), (z+1, ic), (z, tc)]:
                coordinates = {"T": 0, "Z": depth, "C": channel}
                page = int(np.ravel_multi_index(tuple(coordinates[a] for a in axes[:-2]), source.shape[:-2]))
                if tuple(source.pages[page].shape) != tuple(source.shape[-2:]):
                    raise ValueError("selected test page is not scalar YX")
        verified.append((field, path, receipt, info, z))
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".heldout-", dir=output_dir.parent) as temp:
        stage = Path(temp) / "result"
        stage.mkdir()
        scores = {name: [] for name in [*models, "constant"]}
        artifacts, provenance = [], []
        for index, (field, path, receipt, info, z) in enumerate(verified):
            pair = load_context(path, z=z, input_channel=protocol["input_channel"],
                                target_channel=protocol["target_channel"])
            context = scale_plane(pair.input, protocol["scaling"]["input"])
            target = scale_plane(pair.target, protocol["scaling"]["target"])
            files = {}
            def save(name, array):
                filename = f"{index:03d}-{name}.npy"
                np.save(stage / filename, array, allow_pickle=False)
                files[name] = {"file": filename, "sha256": _hash(stage / filename)}
            save("raw-context", pair.input)
            save("raw-target", pair.target)
            save("input", context)
            save("target", target)
            bright = target >= 1.0
            base_record = {"source_id": field["source_id"], "group_id": field["group_id"],
                           "split": "test", "z": z, "bright_pixel_count": int(bright.sum())}
            for name in scores:
                inputs = np.repeat(context[1:2], 3, axis=0) if name == "repeated-centre" else context
                with torch.inference_mode():
                    prediction = (np.full_like(target, protocol["constant_value"]) if name == "constant" else
                                  models[name](input_tensor(inputs))[0, 0].numpy())
                if not np.isfinite(prediction).all():
                    raise ValueError("non-finite held-out prediction")
                save(name, prediction)
                record = base_record | image_scores(prediction, target) | {
                    "bright_pixel_mae": float(np.abs(prediction[bright] - target[bright]).mean()) if bright.any() else None,
                    "prediction_std": float(prediction.std()), "target_std": float(target.std())}
                scores[name].append(record)
                preview = stage / f"{index:03d}-{name}"
                preview.mkdir()
                _validation_preview((base_record, inputs, target), prediction, protocol["constant_value"], preview)
                (preview / "validation-preview.png").rename(preview / "test-preview.png")
            artifacts.append(base_record | {"arrays": files})
            provenance.append({"source_id": field["source_id"], "original_receipt": receipt,
                               "version_id": field["version_id"], "tiff": info, "planes": pair.metadata})
        summaries = {}
        for name, records in scores.items():
            bright_records = [r for r in records if r["bright_pixel_mae"] is not None]
            summaries[name] = summarise_scores(records) | {
                "bright_pixel_mae": summarise_scores([r | {"mae": r["bright_pixel_mae"]}
                                                     for r in bright_records])["mae"],
                "undefined_bright_fields": len(records) - len(bright_records)}
        try:
            revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            revision = "unknown"
        result = {"schema": "locked-heldout-evaluation-v1", "protocol_sha256": identity,
                  "protocol": protocol, "original_receipts_sha256": _hash(receipts_path),
                  "test_pixels_loaded": True, "device": "cpu", "python": platform.python_version(),
                  "torch": torch.__version__, "numpy": np.__version__, "evaluation_source": revision,
                  "packages": {p: importlib.metadata.version(p) for p in
                               ("torch", "numpy", "tifffile", "imagecodecs", "scikit-image", "matplotlib")},
                  "torch_threads": torch.get_num_threads(),
                  "selection": "all locked test fields; no example selection or test-driven tuning",
                  "validation_selected_context": protocol["validation_selected_context"],
                  "summaries": summaries, "fields": scores, "artifacts": artifacts, "provenance": provenance,
                  "limitations": ["Only one held-out plate; no broad generalisation or significance claim.",
                                  "Registration and biological independence remain unverified.",
                                  "Separate CPU evidence, not the user's Mac/MPS run.",
                                  "Fixed [0,1] previews clip display only; targets/predictions and scores remain unclipped."]}
        (stage / "evaluation.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        if output_dir.exists():
            raise ValueError("evaluation output already exists")
        stage.rename(output_dir)
    return {k: result[k] for k in ("protocol_sha256", "test_pixels_loaded", "summaries", "validation_selected_context")}
