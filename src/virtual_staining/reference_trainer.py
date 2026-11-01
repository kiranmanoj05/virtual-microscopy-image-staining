"""Reference-architecture 3D training with complete tiled development validation."""

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from .debug_preview import patch_scores
from .third_party.allen_fnet import Net
from .volume_data import load_volumes


def make_model(width=32, depth=4, initialisation="pytorch-default"):
    if width < 1 or depth not in (1, 2, 3, 4):
        raise ValueError("positive width and depth 1..4 required")
    if initialisation not in ("pytorch-default", "allen-normal"):
        raise ValueError("unknown initialisation")
    model = Net(depth=depth, mult_chan=width)
    if initialisation == "allen-normal":
        # Same distributions and bias handling as release_1/fnet/fnet_model.py.
        with torch.no_grad():
            for module in model.modules():
                if isinstance(module, (torch.nn.Conv3d, torch.nn.ConvTranspose3d)):
                    module.weight.normal_(0.0, 0.02)
                elif isinstance(module, torch.nn.BatchNorm3d):
                    module.weight.normal_(1.0, 0.02)
                    module.bias.zero_()
    return model


def volume_batch(fields, rng, shape, batch_size):
    inputs, targets, positions = [], [], []
    for _ in range(batch_size):
        f, image, target = fields[int(rng.integers(len(fields)))]
        if any(s > n for s, n in zip(shape, image.shape)):
            raise ValueError("patch must fit every training volume")
        start = [int(rng.integers(n - s + 1)) for n, s in zip(image.shape, shape)]
        crop = tuple(slice(a, a + s) for a, s in zip(start, shape))
        inputs.append(np.array(image[crop], copy=True))
        targets.append(np.array(target[crop], copy=True))
        positions.append([f["source_id"], *start])
    return (torch.from_numpy(np.stack(inputs)[:, None]),
            torch.from_numpy(np.stack(targets)[:, None]), positions)


def predict_volume(model, volume, device="cpu", *, core=(32, 64, 64), halo=16, divisor=16):
    """Bound memory, retain all voxels, and never use target data for inference."""
    if (volume.ndim != 3 or len(core) != 3 or min(core) < 1 or halo < 0 or divisor < 1 or
            not np.isfinite(volume).all()):
        raise ValueError("finite ZYX volume and valid tiling geometry required")
    padded = np.pad(volume, [(halo, halo)] * 3, mode="edge")
    output = np.empty(volume.shape, dtype=np.float32)
    model.eval()
    with torch.inference_mode():
        for z in range(0, volume.shape[0], core[0]):
            for y in range(0, volume.shape[1], core[1]):
                for x in range(0, volume.shape[2], core[2]):
                    start = (z, y, x)
                    size = tuple(min(c, n - a) for c, n, a in zip(core, volume.shape, start))
                    crop = tuple(slice(a, a + n + 2 * halo) for a, n in zip(start, size))
                    tile = padded[crop]
                    extra = [(0, (-n) % divisor) for n in tile.shape]
                    tile = np.pad(tile, extra, mode="edge")
                    prediction = model(torch.from_numpy(tile.copy())[None, None].to(device))[0, 0]
                    prediction = prediction.cpu().numpy()
                    if not np.isfinite(prediction).all():
                        raise ValueError("non-finite prediction")
                    centre = tuple(slice(halo, halo + n) for n in size)
                    destination = tuple(slice(a, a + n) for a, n in zip(start, size))
                    output[destination] = prediction[centre]
    return output


def validate_volumes(model, fields, device, depth):
    records, predictions = [], []
    for field, image, target in fields:
        prediction = predict_volume(model, image, device, divisor=2 ** depth)
        scores = patch_scores(prediction, target)
        records.append({"source_id": field["source_id"], "group_id": field["group_id"],
                        **scores, "zero_baseline_mae": float(np.abs(target).mean()),
                        "prediction_std": float(prediction.std()), "target_std": float(target.std())})
        predictions.append(prediction)
    grouped = defaultdict(list)
    for record in records:
        grouped[record["group_id"]].append(record)
    summary = {"fields": len(records), "groups": len(grouped),
               "undefined_pearson_fields": sum(r["pearson"] is None for r in records)}
    for metric in ("mae", "pearson", "zero_baseline_mae"):
        values = [np.mean([r[metric] for r in rows if r[metric] is not None])
                  for rows in grouped.values() if any(r[metric] is not None for r in rows)]
        summary[metric] = float(np.mean(values)) if values else None
    return summary, records, predictions


def train_reference(dataset_path, output_dir, *, steps=1000, width=8, depth=3,
                    patch_shape=(16, 64, 64), batch_size=2, validate_every=250,
                    seed=17, device="cpu", initialisation="allen-normal"):
    if output_dir.exists():
        raise ValueError("reference run output already exists")
    if (len(patch_shape) != 3 or min(steps, batch_size, validate_every) < 1 or
            depth not in (1, 2, 3, 4) or any(s < 2 ** depth or s % 2 ** depth for s in patch_shape)):
        raise ValueError("positive settings and patch dimensions divisible by model stride required")
    if device not in ("cpu", "cuda"):
        raise ValueError("reference baseline currently supports CPU/CUDA; MPS 3D is not verified")
    _, identity, data = load_volumes(dataset_path)
    training = [d for d in data if d[0]["split"] == "train"]
    validation = [d for d in data if d[0]["split"] == "validation"]
    if any(any(s > n for s, n in zip(patch_shape, d[1].shape)) for d in training):
        raise ValueError("patch must fit every training volume")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = make_model(width, depth, initialisation).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.5, 0.999))
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = "unknown"
    config = {"steps": steps, "width": width, "depth": depth, "patch_shape_zyx": list(patch_shape),
              "batch_size": batch_size, "seed": seed, "validate_every": validate_every,
              "device": device, "learning_rate": 0.001, "adam_betas": [0.5, 0.999],
              "loss": "mse", "initialisation": initialisation,
              "validation_selection": "group-balanced full-volume Pearson",
              "prediction_core": [32, 64, 64], "prediction_halo": 16,
              "volume_metadata_sha256": identity, "git_revision": revision,
              "architecture_matches_reference": width == 32 and depth == 4,
              "published_result_reproduced": False,
              "parameter_count": sum(p.numel() for p in model.parameters()),
              "python": platform.python_version(), "threads": torch.get_num_threads(),
              "packages": {p: importlib.metadata.version(p) for p in ("torch", "numpy", "scipy")}}
    output_dir.mkdir(parents=True)
    (output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    history, losses, trace = [], [], hashlib.sha256()
    best, best_step, started = -float("inf"), None, time.perf_counter()

    def save(name, step):
        torch.save({"model_state": model.state_dict(), "optimizer_state": optimiser.state_dict(),
                    "config": config, "step": step, "numpy_rng_state": rng.bit_generator.state,
                    "torch_rng_state": torch.get_rng_state(), "history": history}, output_dir / name)

    for step in range(1, steps + 1):
        model.train()
        inputs, targets, positions = volume_batch(training, rng, patch_shape, batch_size)
        trace.update(json.dumps(positions, separators=(",", ":")).encode())
        optimiser.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(model(inputs.to(device)), targets.to(device))
        if not torch.isfinite(loss):
            raise ValueError("non-finite training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"), error_if_nonfinite=True, foreach=False)
        optimiser.step()
        losses.append(float(loss.detach()))
        if step % 25 == 0:
            print(f"Reference step {step}/{steps}: MSE={losses[-1]:.5f}", file=sys.stderr, flush=True)
        if step % validate_every == 0 or step == steps:
            summary, records, _ = validate_volumes(model, validation, device, depth)
            history.append({"step": step, "training_mse": losses[-1], "validation": summary,
                            "fields": records, "elapsed_seconds": time.perf_counter() - started})
            if summary["pearson"] is not None and summary["pearson"] > best:
                best, best_step = summary["pearson"], step
                save("best-model.pt", step)
            save("last-model.pt", step)
            (output_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
            print(f"Full-volume validation: {summary}", file=sys.stderr, flush=True)
    if best_step is None:
        raise ValueError("all validation correlations undefined; no model selected")
    saved = torch.load(output_dir / "best-model.pt", map_location=device, weights_only=True)
    model.load_state_dict(saved["model_state"])
    summary, records, predictions = validate_volumes(model, validation, device, depth)
    for index, (item, prediction) in enumerate(zip(validation, predictions)):
        np.save(output_dir / f"validation-{index}-prediction.npy", prediction, allow_pickle=False)
        validation_gallery(item, prediction, output_dir / f"validation-{index}.png")
    result = {"scope": "development pilot only; old test excluded; no new final test result",
              "target_units": "relative per-volume fluorescence, not calibrated raw intensity",
              "config": config, "best_step": best_step, "best_validation": summary,
              "best_fields": records, "history": history, "training_loss_history": losses,
              "elapsed_seconds": time.perf_counter() - started,
              "sampling_trace_sha256": trace.hexdigest(), "test_pixels_loaded": False,
              "checkpoints": {name: hashlib.sha256((output_dir / name).read_bytes()).hexdigest()
                              for name in ("best-model.pt", "last-model.pt")}}
    (output_dir / "training.json").write_text(json.dumps(result, indent=2) + "\n")
    return {"scope": result["scope"], "best_step": best_step, "best_validation": summary,
            "elapsed_seconds": result["elapsed_seconds"], "output_dir": str(output_dir)}


def validation_gallery(item, prediction, output_path):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    field, inputs, target = item
    figure = Figure(figsize=(15, 10), constrained_layout=True)
    FigureCanvasAgg(figure)
    for axes, q in zip(figure.subplots(3, 4), (0.25, 0.5, 0.75)):
        z = int((inputs.shape[0] - 1) * q)
        panels = [(inputs[z], "Brightfield", "gray", -2, 2),
                  (target[z], "Measured DNA", "magma", -1, 3),
                  (prediction[z], "Predicted DNA", "magma", -1, 3),
                  (np.abs(target[z] - prediction[z]), "Absolute error", "inferno", 0, 3)]
        for axis, (a, title, cmap, low, high) in zip(axes, panels):
            axis.imshow(a, cmap=cmap, vmin=low, vmax=high, interpolation="nearest")
            axis.set_title(f"{title}; z={z}")
            axis.set_axis_off()
    figure.suptitle(f"{field['source_id']} · DEVELOPMENT ONLY · standardised units; fixed ranges")
    figure.savefig(output_path, dpi=110)
