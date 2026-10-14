"""Train a small pilot with whole-plane validation and training-derived baselines."""

import hashlib
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from skimage.metrics import structural_similarity

from .debug_preview import patch_scores
from .model import UNet
from .training_data import scale_plane


def load_development_data(path: Path, *, scaled=True):
    metadata = json.loads(path.read_text())
    identity = metadata.pop("prepared_metadata_sha256")
    canonical = json.dumps(metadata, sort_keys=True, separators=(",", ":"))
    if hashlib.sha256(canonical.encode()).hexdigest() != identity:
        raise ValueError("prepared dataset fingerprint mismatch")
    if metadata["test_pixels_loaded"] or metadata["scaling"]["fitted_on"] != "train only":
        raise ValueError("expected unopened test pixels and training-only scaling")
    data, seen, groups = [], set(), {}
    for field in metadata["fields"]:
        if field["split"] not in ("train", "validation"):
            raise ValueError("trainer accepts development fields only")
        if field["source_id"] in seen:
            raise ValueError("duplicate prepared source")
        seen.add(field["source_id"])
        previous = groups.setdefault(field["group_id"], field["split"])
        if previous != field["split"]:
            raise ValueError("prepared group crosses splits")
        arrays = []
        for name in ("input", "target"):
            filename = field[name + "_file"]
            if Path(filename).name != filename:
                raise ValueError("prepared files must be local basenames")
            source = path.parent / filename
            if hashlib.sha256(source.read_bytes()).hexdigest() != field[name + "_sha256"]:
                raise ValueError("prepared pixels failed checksum verification")
            array = np.load(source, allow_pickle=False)
            if array.ndim != 2 or min(array.shape) < 8 or not np.isfinite(array).all():
                raise ValueError("expected a finite 2D plane with dimensions >= 8")
            arrays.append(scale_plane(array, metadata["scaling"][name]) if scaled else array)
        if arrays[0].shape != arrays[1].shape:
            raise ValueError("paired planes have different shapes")
        data.append((field, *arrays))
    if not all(any(f[0]["split"] == split for f in data) for split in ("train", "validation")):
        raise ValueError("both training and validation fields are required")
    return metadata, identity, data


def sample_batch(data, rng, patch_size, batch_size):
    inputs, targets = [], []
    for _ in range(batch_size):
        _, image, target = data[int(rng.integers(len(data)))]
        if min(image.shape) < patch_size:
            raise ValueError("training patch does not fit source plane")
        y = int(rng.integers(image.shape[0] - patch_size + 1))
        x = int(rng.integers(image.shape[1] - patch_size + 1))
        pair = [a[y:y + patch_size, x:x + patch_size] for a in (image, target)]
        rotation = int(rng.integers(4))
        pair = [np.rot90(a, rotation) for a in pair]
        for axis in (0, 1):
            if rng.integers(2):
                pair = [np.flip(a, axis) for a in pair]
        inputs.append(pair[0].copy())
        targets.append(pair[1].copy())
    return torch.from_numpy(np.stack(inputs)[:, None]), torch.from_numpy(np.stack(targets)[:, None])


def image_scores(prediction, target):
    return patch_scores(prediction, target) | {
        "ssim": float(structural_similarity(target, prediction, data_range=1.0, win_size=7))}


def summarise_scores(records):
    groups = defaultdict(list)
    for record in records:
        groups[record["group_id"]].append(record)
    result = {"field_count": len(records), "group_count": len(groups),
              "undefined_pearson_fields": sum(r["pearson"] is None for r in records)}
    for metric in ("mae", "pearson", "ssim"):
        per_group = [np.mean([r[metric] for r in rows if r[metric] is not None])
                     for rows in groups.values() if any(r[metric] is not None for r in rows)]
        result[metric] = float(np.mean(per_group)) if per_group else None
    return result


def validate(model, data, device, *, constant=None):
    records, predictions = [], []
    model.eval()
    with torch.no_grad():
        for field, inputs, target in data:
            prediction = (np.full_like(target, constant) if constant is not None else
                          model(torch.from_numpy(inputs.copy())[None, None].to(device))[0, 0].cpu().numpy())
            records.append({"source_id": field["source_id"], "group_id": field["group_id"],
                            "z": field["z"], **image_scores(prediction, target)})
            predictions.append(prediction)
    return summarise_scores(records), records, predictions


def train_pilot(dataset_path: Path, output_dir: Path, *, steps=500, batch_size=2,
                patch_size=128, width=16, seed=17, validate_every=100, device_name="auto",
                resume_path: Path | None = None):
    if min(steps, batch_size, width, validate_every) < 1 or patch_size < 8:
        raise ValueError("positive settings and patch_size >= 8 are required")
    if output_dir.exists():
        raise ValueError("training run output already exists")
    metadata, identity, data = load_development_data(dataset_path)
    training = [d for d in data if d[0]["split"] == "train"]
    validation = [d for d in data if d[0]["split"] == "validation"]
    if any(min(d[1].shape) < patch_size for d in training):
        raise ValueError("training patch does not fit every training plane")
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    if device_name not in ("cpu", "mps", "cuda"):
        raise ValueError("unsupported device")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = UNet(width).to(device_name)
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
    constant = float(np.median(np.concatenate([d[2].ravel() for d in training])))
    baseline, baseline_fields, _ = validate(model, validation, device_name, constant=constant)
    untrained, untrained_fields, _ = validate(model, validation, device_name)
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                           stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = "unknown"
    config = {"steps": steps, "batch_size": batch_size, "patch_size": patch_size,
              "width": width, "seed": seed, "validate_every": validate_every,
              "device": device_name, "learning_rate": 1e-3,
              "primary_metric": "group-balanced whole-plane validation MAE",
              "ssim_data_range_scaled": 1.0, "ssim_window": 7,
              "prepared_metadata_sha256": identity, "pilot_plan_sha256": metadata["pilot_plan_sha256"],
              "git_revision": revision, "depth_policy": metadata["depth_policy"],
              "python": platform.python_version(),
              "packages": {p: importlib.metadata.version(p) for p in
                           ("torch", "numpy", "tifffile", "imagecodecs", "scikit-image")}}
    start_step = 0
    history = [{"step": 0, "training_loss": None, "validation": untrained}]
    best, best_step, best_records = untrained["mae"], 0, untrained_fields
    if resume_path is not None:
        if resume_path.name != "last-model.pt":
            raise ValueError("resume requires last-model.pt from a completed run")
        previous = json.loads((resume_path.parent / "training.json").read_text())
        if hashlib.sha256(resume_path.read_bytes()).hexdigest() != previous["last_checkpoint_sha256"]:
            raise ValueError("resume checkpoint checksum mismatch")
        resumed = torch.load(resume_path, map_location="cpu", weights_only=True)
        for key in ("prepared_metadata_sha256", "pilot_plan_sha256", "seed", "patch_size", "width", "batch_size", "learning_rate", "validate_every"):
            if resumed["config"][key] != config[key]:
                raise ValueError(f"resume setting mismatch: {key}")
        start_step = resumed["step"]
        if steps <= start_step:
            raise ValueError("total steps must exceed the checkpoint step")
        old_best = resume_path.parent / "best-model.pt"
        if hashlib.sha256(old_best.read_bytes()).hexdigest() != previous["best_checkpoint_sha256"]:
            raise ValueError("previous best checkpoint checksum mismatch")
        model.load_state_dict(resumed["model_state"])
        optimiser.load_state_dict(resumed["optimizer_state"])
        rng.bit_generator.state = resumed["numpy_rng_state"]
        torch.set_rng_state(resumed["torch_rng_state"].cpu())
        history = resumed["history"]
        best, best_step, best_records = previous["best_model"]["mae"], previous["best_step"], previous["best_fields"]
        config["resume_checkpoint_sha256"] = previous["last_checkpoint_sha256"]
        config["resume_from_step"] = start_step
    output_dir.mkdir(parents=True)
    (output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    def checkpoint(step, name):
        torch.save({"model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    "optimizer_state": optimiser.state_dict(), "step": step, "config": config,
                    "scaling": metadata["scaling"], "numpy_rng_state": rng.bit_generator.state,
                    "torch_rng_state": torch.get_rng_state(), "history": history}, output_dir / name)

    if resume_path is None:
        checkpoint(0, "best-model.pt")
    else:
        shutil.copyfile(resume_path.parent / "best-model.pt", output_dir / "best-model.pt")
    started = time.perf_counter()
    for step in range(start_step + 1, steps + 1):
        model.train()
        inputs, targets = (a.to(device_name) for a in sample_batch(training, rng, patch_size, batch_size))
        optimiser.zero_grad(set_to_none=True)
        loss = torch.nn.functional.l1_loss(model(inputs), targets)
        if not torch.isfinite(loss):
            raise ValueError("non-finite training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"), error_if_nonfinite=True, foreach=False)
        optimiser.step()
        if step % validate_every == 0 or step == steps:
            scores, records, predictions = validate(model, validation, device_name)
            history.append({"step": step, "training_loss": float(loss.item()), "validation": scores})
            if scores["mae"] < best:
                best, best_step, best_records = scores["mae"], step, records
                checkpoint(step, "best-model.pt")
                for index, prediction in enumerate(predictions):
                    np.save(output_dir / f"best-validation-{index}.npy", prediction, allow_pickle=False)
            print(f"Step {step}/{steps}: train L1={loss.item():.5f}, val MAE={scores['mae']:.5f}, "
                  f"constant MAE={baseline['mae']:.5f}", file=sys.stderr, flush=True)
            (output_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    checkpoint(steps, "last-model.pt")
    elapsed = time.perf_counter() - started
    saved = torch.load(output_dir / "best-model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(saved["model_state"])
    best_scores, best_records, predictions = validate(model, validation, device_name)
    for index, prediction in enumerate(predictions):
        np.save(output_dir / f"best-validation-{index}.npy", prediction, allow_pickle=False)
    _validation_preview(validation[0], predictions[0], constant, output_dir)
    result = {"scope": "small pilot validation only; test set untouched", "config": config,
              "train_fields": len(training), "validation_fields": len(validation),
              "constant_baseline": baseline, "constant_baseline_fields": baseline_fields,
              "constant_definition": "median of all training middle-plane target pixels", "constant_value": constant,
              "untrained_model": untrained, "best_model": best_scores, "best_step": best_step,
              "best_fields": best_records, "beats_constant_mae": best < baseline["mae"],
              "elapsed_seconds": elapsed, "steps_this_run": steps - start_step,
              "seconds_per_step_including_validation": elapsed / (steps - start_step),
              "best_checkpoint_sha256": hashlib.sha256((output_dir / "best-model.pt").read_bytes()).hexdigest(),
              "last_checkpoint_sha256": hashlib.sha256((output_dir / "last-model.pt").read_bytes()).hexdigest(),
              "test_pixels_loaded": False, "output_dir": str(output_dir)}
    (output_dir / "training.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def _validation_preview(item, prediction, constant, output_dir):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    field, inputs, target = item
    figure = Figure(figsize=(18, 4), constrained_layout=True)
    FigureCanvasAgg(figure)
    panels = [(inputs, "Brightfield", "gray"), (target, "Measured DNA", "magma"),
              (prediction, f"{field['split'].capitalize()} prediction", "magma"),
              (np.full_like(target, constant), "Training-derived median", "magma"),
              (np.abs(prediction - target), "Absolute error", "inferno")]
    for axis, (array, title, cmap) in zip(figure.subplots(1, 5), panels):
        axis.imshow(array, cmap=cmap, vmin=0, vmax=1)
        axis.set_title(title)
        axis.set_axis_off()
    figure.suptitle(f"{field['source_id']}, z={field['z']} · full-plane {field['split']}, fixed [0, 1] display range")
    figure.savefig(output_dir / "validation-preview.png", dpi=150)
