"""Render measured paired planes for inspection; never fit training scaling."""

import json
from pathlib import Path

import numpy as np

from .paired_planes import load_pair


def preview_pair(path: Path, *, z_indices: list[int], input_channel: int,
                 target_channel: int, out_dir: Path) -> dict:
    if not z_indices or len(z_indices) > 8:
        raise ValueError("select between one and eight z planes for a preview")
    pairs = [load_pair(path, z=z, input_channel=input_channel, target_channel=target_channel)
             for z in z_indices]
    bounds = {}
    for role in ("input", "target"):
        values = np.concatenate([getattr(pair, role).ravel() for pair in pairs])
        low, high = np.percentile(values, [1, 99])
        bounds[role] = [float(low), float(high)]

    def display(array, role):
        low, high = bounds[role]
        if high <= low:
            return np.zeros(array.shape, dtype=np.float32)
        return np.clip((array.astype(np.float32) - low) / (high - low), 0, 1)

    # Use a headless figure API, so exporting works on Macs and in CI.
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(12, 3.4 * len(pairs)), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots(len(pairs), 3, squeeze=False)
    for row, pair in enumerate(pairs):
        brightfield = display(pair.input, "input")
        dna = display(pair.target, "target")
        overlay = np.repeat((0.65 * brightfield)[..., None], 3, axis=-1)
        overlay[..., 1] = np.clip(overlay[..., 1] + 0.7 * dna, 0, 1)
        axes[row, 0].imshow(brightfield, cmap="gray", vmin=0, vmax=1)
        axes[row, 1].imshow(dna, cmap="magma", vmin=0, vmax=1)
        axes[row, 2].imshow(overlay, vmin=0, vmax=1)
        for column, title in enumerate(("Measured input", "Measured target", "Target in green over input")):
            axes[row, column].set_title(f"z={pair.metadata['z']} · {title}")
            axes[row, column].set_axis_off()
    figure.suptitle(
        f"Paired measurements · input channel {input_channel}, target channel {target_channel}\n"
        "Shared per-channel display ranges across selected depths; no model prediction"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / "preview.png"
    figure.savefig(png, dpi=150)
    report = {
        "source_path": str(path), "preview_path": str(png),
        "input_channel": input_channel, "target_channel": target_channel,
        "z_indices": z_indices, "pages_decoded": 2 * len(pairs),
        "pairs": [pair.metadata for pair in pairs],
        "display_scaling": {"purpose": "visual inspection only; not training preprocessing",
                            "percentiles": [1, 99], "shared_across_selected_z": True,
                            "input_range": bounds["input"], "target_range": bounds["target"]},
        "registration_status": "unverified; visual preview is not proof of registration",
        "predictions_generated": False,
    }
    (out_dir / "preview.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
