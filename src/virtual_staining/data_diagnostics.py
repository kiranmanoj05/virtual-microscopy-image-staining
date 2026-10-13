"""Describe raw development intensities without a checkpoint or test access."""

import json
from pathlib import Path

import numpy as np

from .diagnostics import crop_locations, intensity_summary
from .trainer import load_development_data
from .training_data import scale_plane


def channel_summary(raw, params):
    # Keep the pre-clipping distribution: the model's clipped input alone can
    # conceal a shift outside the training-fitted intensity interval.
    unbounded = scale_plane(raw, params | {"clip": False})
    return {"raw": intensity_summary(raw),
            "scaled_before_clipping": intensity_summary(unbounded),
            "model_values": intensity_summary(scale_plane(raw, params)),
            "fraction_below_training_low": float((raw < params["low"]).mean()),
            "fraction_above_training_high": float((raw > params["high"]).mean())}


def signal_summary(target, constant):
    bright = target >= 1.0
    errors = np.abs(target - constant)
    total = float(errors.sum(dtype=np.float64))
    return {"constant_mae": float(errors.mean()),
            "bright_pixel_count": int(bright.sum()),
            "bright_pixel_fraction": float(bright.mean()),
            "bright_constant_mae": float(errors[bright].mean()) if bright.any() else None,
            "bright_fraction_of_constant_absolute_error":
                float(errors[bright].sum(dtype=np.float64) / total) if total else None}


def diagnose_data(dataset_path: Path, output_dir: Path, *, patch_size=128):
    if output_dir.exists():
        raise ValueError("data diagnostic output already exists")
    if patch_size < 8:
        raise ValueError("diagnostic patch_size must be >= 8")
    metadata, identity, data = load_development_data(dataset_path, scaled=False)
    if any(min(d[1].shape) < patch_size for d in data):
        raise ValueError("diagnostic patch does not fit every development plane")
    scaling = metadata["scaling"]
    train_targets = [scale_plane(target, scaling["target"]) for field, _, target in data
                     if field["split"] == "train"]
    constant = float(np.median(np.concatenate([a.ravel() for a in train_targets])))
    records = []
    for field, inputs, raw_target in data:
        target = scale_plane(raw_target, scaling["target"])
        patches = []
        for y, x in crop_locations(target.shape, patch_size):
            crop = target[y:y + patch_size, x:x + patch_size]
            patches.append({"y": y, "x": x, "height": patch_size, "width": patch_size,
                            "target_intensities_scaled": intensity_summary(crop),
                            **signal_summary(crop, constant)})
        records.append({"source_id": field["source_id"], "group_id": field["group_id"],
                        "split": field["split"], "z": field["z"],
                        "input": channel_summary(inputs, scaling["input"]),
                        "target": channel_summary(raw_target, scaling["target"]),
                        **signal_summary(target, constant), "fixed_patches": patches})
    # Descriptive summaries weight fields equally within groups and groups
    # equally within a split, rather than pooling pixels across acquisition groups.
    splits = {}
    for split in ("train", "validation"):
        rows = [r for r in records if r["split"] == split]
        groups = sorted({r["group_id"] for r in rows})
        metrics = {}
        for metric in ("constant_mae", "bright_pixel_fraction",
                       "bright_fraction_of_constant_absolute_error"):
            values = []
            for group in groups:
                observations = [r[metric] for r in rows
                                if r["group_id"] == group and r[metric] is not None]
                if observations:
                    values.append(float(np.mean(observations)))
            metrics[metric] = float(np.mean(values)) if values else None
        splits[split] = {"field_count": len(rows), "group_count": len(groups),
                         "group_balanced": metrics}
    result = {"purpose": "descriptive development data diagnostics; no model or selection",
              "prepared_metadata_sha256": identity,
              "pilot_plan_sha256": metadata["pilot_plan_sha256"],
              "scaling": scaling, "constant_value_scaled": constant,
              "constant_definition": "median of training target pixels only",
              "bright_definition": "target >= fixed training-target 99th percentile",
              "patch_policy": "fixed centre and four corners; descriptive, not training sampling",
              "patch_size": patch_size, "split_summaries": splits, "fields": records,
              "test_pixels_loaded": False,
              "limitations": ["Intensity differences do not establish their cause.",
                              "Fixed crops do not estimate random training crop prevalence.",
                              "No registration or model performance is established."]}
    output_dir.mkdir(parents=True)
    (output_dir / "data-diagnostics.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return {k: v for k, v in result.items() if k != "fields"} | {"output_dir": str(output_dir)}
