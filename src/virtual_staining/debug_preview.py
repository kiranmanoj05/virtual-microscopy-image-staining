"""Inspect a trained debug patch against its measured signal and constant baseline."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .model import UNet
from .overfit import load_training_patch


def patch_scores(prediction: np.ndarray, target: np.ndarray) -> dict:
    if prediction.shape != target.shape or not np.isfinite(prediction).all() or not np.isfinite(target).all():
        raise ValueError("metrics require finite arrays with matching shapes")
    p, t = prediction.astype(np.float64).ravel(), target.astype(np.float64).ravel()
    p, t = p - p.mean(), t - t.mean()
    denominator = float(np.linalg.norm(p) * np.linalg.norm(t))
    return {"mae": float(np.abs(prediction.astype(np.float64) - target).mean()),
            "pearson": float(np.dot(p, t) / denominator) if denominator > 0 else None}


def preview_overfit(dataset_path: Path, checkpoint_path: Path, output_dir: Path) -> dict:
    if output_dir.exists():
        raise ValueError("debug preview output already exists")
    run = json.loads((checkpoint_path.parent / "overfit.json").read_text(encoding="utf-8"))
    checkpoint_hash = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    if checkpoint_hash != run["checkpoint_sha256"]:
        raise ValueError("debug checkpoint failed checksum verification")
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = saved["config"]
    metadata, identity, field, arrays = load_training_patch(dataset_path, config["patch_size"])
    if (identity != config["prepared_metadata_sha256"] or
            metadata["pilot_plan_sha256"] != config["pilot_plan_sha256"] or
            field["source_id"] != config["source_id"] or field["z"] != config["z"] or
            metadata["scaling"] != saved["scaling"]):
        raise ValueError("checkpoint does not match the prepared training patch")
    model = UNet(width=config["width"])
    model.load_state_dict(saved["model_state"], strict=True)
    model.eval()
    with torch.no_grad():
        prediction = model(torch.from_numpy(arrays[0].copy())[None, None])[0, 0].numpy()
    target = arrays[1]
    constant = np.full_like(target, np.median(target))
    model_scores, baseline_scores = patch_scores(prediction, target), patch_scores(constant, target)
    scale = metadata["scaling"]["target"]["high"] - metadata["scaling"]["target"]["low"]
    model_scores["mae_raw_intensity_units"] = model_scores["mae"] * scale
    baseline_scores["mae_raw_intensity_units"] = baseline_scores["mae"] * scale
    result = {
        "scope": "same training patch used for optimisation; not held-out performance",
        "source_id": field["source_id"], "z": field["z"], "patch_size": config["patch_size"],
        "prepared_metadata_sha256": identity, "checkpoint_sha256": checkpoint_hash,
        "inference_device": "cpu", "model_scores": model_scores, "constant_scores": baseline_scores,
        "constant_definition": "median of this training patch target (optimal constant for L1)",
        "beats_constant_mae": model_scores["mae"] < baseline_scores["mae"],
        "display_range_scaled": [0, 1], "target_values_clipped_for_metrics": False,
        "prediction_range_scaled": [float(prediction.min()), float(prediction.max())],
        "target_range_scaled": [float(target.min()), float(target.max())],
        "test_pixels_loaded": False,
    }
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    figure = Figure(figsize=(12, 8), constrained_layout=True)
    FigureCanvasAgg(figure)
    axes = figure.subplots(2, 3)
    panels = [(axes[0, 0], arrays[0], "Brightfield input", "gray"),
              (axes[0, 1], target, "Measured DNA fluorescence", "magma"),
              (axes[0, 2], prediction, "U-Net prediction: training patch", "magma"),
              (axes[1, 0], constant, "Same-patch median baseline", "magma"),
              (axes[1, 1], np.abs(prediction - target), "Absolute prediction error", "inferno")]
    for axis, array, title, cmap in panels:
        axis.imshow(array, cmap=cmap, vmin=0, vmax=1)
        axis.set_title(title)
        axis.set_axis_off()
    axes[1, 2].plot(np.arange(1, len(run["loss_history"]) + 1), run["loss_history"])
    axes[1, 2].axhline(baseline_scores["mae"], color="orange", linestyle="--", label="median baseline")
    axes[1, 2].set(xlabel="Optimisation step", ylabel="Scaled L1 error", title="One-patch learning check")
    axes[1, 2].legend()
    figure.suptitle(f"{field['source_id']}, z={field['z']} · training debug only\n"
                    "Signal and error panels use fixed [0, 1] display scales; metrics use unclipped values")
    output_dir.mkdir(parents=True)
    figure.savefig(output_dir / "preview.png", dpi=150)
    np.save(output_dir / "prediction-scaled.npy", prediction, allow_pickle=False)
    (output_dir / "preview.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result | {"output_dir": str(output_dir)}
