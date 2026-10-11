"""Debug gradients and target handling by learning one fixed training patch."""

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from .model import UNet
from .training_data import scale_plane


def overfit_patch(dataset_path: Path, output_dir: Path, *, steps: int = 100,
                  patch_size: int = 128, width: int = 16, seed: int = 17,
                  device_name: str = "auto") -> dict:
    if steps < 1 or patch_size < 8:
        raise ValueError("positive steps and patch_size >= 8 are required")
    if output_dir.exists():
        raise ValueError("overfit output already exists")
    metadata = json.loads(dataset_path.read_text(encoding="utf-8"))
    identity = metadata.pop("prepared_metadata_sha256")
    canonical = json.dumps(metadata, sort_keys=True, separators=(",", ":"))
    if hashlib.sha256(canonical.encode()).hexdigest() != identity:
        raise ValueError("prepared dataset metadata fingerprint mismatch")
    if metadata["scaling"]["fitted_on"] != "train only" or metadata["test_pixels_loaded"]:
        raise ValueError("expected training-only scaling and unopened test pixels")
    field = next((f for f in metadata["fields"] if f["split"] == "train"), None)
    if field is None:
        raise ValueError("no training field in prepared dataset")
    arrays = []
    for name in ("input", "target"):
        filename = field[name + "_file"]
        if Path(filename).name != filename:
            raise ValueError("prepared files must be local basenames")
        path = dataset_path.parent / filename
        if hashlib.sha256(path.read_bytes()).hexdigest() != field[name + "_sha256"]:
            raise ValueError("prepared pixels failed checksum verification")
        array = np.load(path, allow_pickle=False)
        if array.ndim != 2 or min(array.shape) < patch_size or not np.isfinite(array).all():
            raise ValueError("patch does not fit a finite 2D plane")
        y, x = ((size - patch_size) // 2 for size in array.shape)
        arrays.append(scale_plane(array[y:y + patch_size, x:x + patch_size], metadata["scaling"][name]))
    if arrays[0].shape != arrays[1].shape:
        raise ValueError("paired patch shapes disagree")
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    if device_name not in ("cpu", "mps", "cuda"):
        raise ValueError("unsupported device")
    torch.manual_seed(seed)
    model = UNet(width=width).to(device_name)
    inputs, targets = (torch.from_numpy(a.copy())[None, None].to(device_name) for a in arrays)
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.L1Loss()
    with torch.no_grad():
        initial_loss = float(loss_fn(model(inputs), targets).item())
    losses = []
    started = time.perf_counter()
    for step in range(steps):
        optimiser.zero_grad(set_to_none=True)
        loss = loss_fn(model(inputs), targets)
        if not torch.isfinite(loss):
            raise ValueError("non-finite training loss")
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise ValueError("non-finite gradients")
        optimiser.step()
        losses.append(float(loss.item()))
        if step == 0 or (step + 1) % 10 == 0:
            print(f"Overfit {step + 1}/{steps}: L1={losses[-1]:.6f}", file=sys.stderr, flush=True)
    with torch.no_grad():
        final_loss = float(loss_fn(model(inputs), targets).item())
    elapsed = time.perf_counter() - started
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL,
                                           text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = "unknown"
    result = {
        "purpose": "one-training-patch debugging; not held-out performance",
        "source_id": field["source_id"], "group_id": field["group_id"], "z": field["z"],
        "pilot_plan_sha256": metadata["pilot_plan_sha256"], "prepared_metadata_sha256": identity,
        "device": device_name, "seed": seed, "steps": steps, "patch_size": patch_size,
        "width": width, "learning_rate": 1e-3, "loss": "L1 on training-scaled target",
        "initial_loss": initial_loss, "final_loss": final_loss, "loss_decreased": final_loss < initial_loss,
        "elapsed_seconds": elapsed, "seconds_per_step": elapsed / steps,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "git_revision": revision, "python": platform.python_version(),
        "packages": {p: importlib.metadata.version(p) for p in ("torch", "numpy", "tifffile", "imagecodecs")},
        "loss_history": losses, "test_pixels_loaded": False,
    }
    output_dir.mkdir(parents=True)
    checkpoint = output_dir / "debug-model.pt"
    torch.save({"model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "config": result, "scaling": metadata["scaling"]}, checkpoint)
    result["checkpoint_sha256"] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    (output_dir / "overfit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return {key: value for key, value in result.items() if key != "loss_history"} | {"output_dir": str(output_dir)}
