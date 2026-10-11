"""Inspect learning, intensity shifts and crop-context effects without test access."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .model import UNet
from .trainer import image_scores, load_development_data, summarise_scores, _validation_preview


def intensity_summary(array):
    return {"min": float(array.min()), "max": float(array.max()),
            "mean": float(array.mean()), "std": float(array.std()),
            "p01": float(np.percentile(array, 1)), "p50": float(np.percentile(array, 50)),
            "p99": float(np.percentile(array, 99)),
            "fraction_le_zero": float((array <= 0).mean()),
            "fraction_ge_one": float((array >= 1).mean())}


def crop_locations(shape, size):
    height, width = shape
    if min(shape) < size:
        raise ValueError("diagnostic patch does not fit the image")
    positions = [((height - size) // 2, (width - size) // 2), (0, 0),
                 (0, width - size), (height - size, 0), (height - size, width - size)]
    return list(dict.fromkeys(positions))


def diagnose_training(dataset_path: Path, run_dir: Path, output_dir: Path, *, device_name="cpu"):
    if output_dir.exists():
        raise ValueError("diagnostic output already exists")
    metadata, identity, data = load_development_data(dataset_path)
    run = json.loads((run_dir / "training.json").read_text())
    if identity != run["config"]["prepared_metadata_sha256"] or run["test_pixels_loaded"]:
        raise ValueError("training run does not match the development dataset")
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    if device_name not in ("cpu", "mps", "cuda"):
        raise ValueError("unsupported diagnostic device")
    constant = float(np.median(np.concatenate([d[2].ravel() for d in data if d[0]["split"] == "train"])))
    if not np.isclose(constant, run["constant_value"], rtol=0, atol=1e-7):
        raise ValueError("training-derived baseline does not match saved run")
    for name in ("best", "last"):
        path = run_dir / f"{name}-model.pt"
        if hashlib.sha256(path.read_bytes()).hexdigest() != run[f"{name}_checkpoint_sha256"]:
            raise ValueError(f"{name} checkpoint checksum mismatch")
    output_dir.mkdir(parents=True)
    output = {"purpose": "development diagnostics; no experiment settings changed",
              "prepared_metadata_sha256": identity, "device": device_name,
              "bright_target_threshold_scaled": 1.0,
              "bright_target_definition": "above the fixed training-target 99th percentile",
              "test_pixels_loaded": False, "checkpoints": {}}
    for name in ("best", "last"):
        saved = torch.load(run_dir / f"{name}-model.pt", map_location="cpu", weights_only=True)
        if saved["config"]["prepared_metadata_sha256"] != identity:
            raise ValueError("checkpoint dataset identity mismatch")
        model = UNet(saved["config"]["width"]).to(device_name)
        model.load_state_dict(saved["model_state"])
        model.eval()
        size = saved["config"]["patch_size"]
        records, previews = [], set()
        with torch.no_grad():
            for field, inputs, target in data:
                prediction = model(torch.from_numpy(inputs.copy())[None, None].to(device_name))[0, 0].cpu().numpy()
                baseline = np.full_like(target, constant)
                bright = target >= 1.0
                patch_results = []
                for y, x in crop_locations(target.shape, size):
                    image_crop = inputs[y:y + size, x:x + size]
                    target_crop = target[y:y + size, x:x + size]
                    full_crop = prediction[y:y + size, x:x + size]
                    isolated = model(torch.from_numpy(image_crop.copy())[None, None].to(device_name))[0, 0].cpu().numpy()
                    patch_results.append({"y": y, "x": x,
                                          "isolated_patch_mae": float(np.abs(isolated - target_crop).mean()),
                                          "full_image_crop_mae": float(np.abs(full_crop - target_crop).mean()),
                                          "context_difference_mae": float(np.abs(full_crop - isolated).mean())})
                scores = image_scores(prediction, target)
                records.append({"source_id": field["source_id"], "group_id": field["group_id"],
                                "split": field["split"], **scores,
                                "constant_scores": image_scores(baseline, target),
                                "input_intensities": intensity_summary(inputs),
                                "target_intensities": intensity_summary(target),
                                "prediction_intensities": intensity_summary(prediction),
                                "bright_pixel_count": int(bright.sum()),
                                "bright_pixel_mae": float(np.abs(prediction[bright] - target[bright]).mean()) if bright.any() else None,
                                "constant_bright_pixel_mae": float(np.abs(baseline[bright] - target[bright]).mean()) if bright.any() else None,
                                "patch_context": patch_results})
                if field["split"] not in previews:
                    preview_dir = output_dir / f"{name}-{field['split']}"
                    preview_dir.mkdir()
                    _validation_preview((field, inputs, target), prediction, constant, preview_dir)
                    # Use a training-specific filename for training previews.
                    if field["split"] == "train":
                        (preview_dir / "validation-preview.png").rename(preview_dir / "training-preview.png")
                    previews.add(field["split"])
        splits = {}
        for split in ("train", "validation"):
            split_records = [r for r in records if r["split"] == split]
            splits[split] = {"model": summarise_scores(split_records),
                             "constant": summarise_scores([r | r["constant_scores"] for r in split_records])}
        output["checkpoints"][name] = {"step": saved["step"], "split_scores": splits, "fields": records}
    (output_dir / "diagnostics.json").write_text(json.dumps(output, indent=2) + "\n")
    return {"test_pixels_loaded": False, "output_dir": str(output_dir),
            "checkpoints": {k: {"step": v["step"], "split_scores": v["split_scores"]}
                            for k, v in output["checkpoints"].items()}}
