"""Read paired planes without loading a complete multi-channel Z-stack."""

from dataclasses import dataclass
from math import prod
from pathlib import Path

import numpy as np
import tifffile


@dataclass
class PlanePair:
    input: np.ndarray
    target: np.ndarray
    metadata: dict


def _load_selected(path: Path, selections, series_index):
    """Validate all selections before decoding scalar pages.

    Supported layouts have scalar YX pages and C plus optional Z/T leading
    axes. T must be singleton. Unsupported packed/volumetric layouts fail
    explicitly instead of falling back to loading the entire image.
    """
    with tifffile.TiffFile(path) as image:
        if not 0 <= series_index < len(image.series):
            raise ValueError("series index is outside this TIFF")
        series = image.series[series_index]
        axes, shape = series.axes, series.shape
        if (not axes.endswith("YX") or "C" not in axes[:-2]
                or len(set(axes)) != len(axes)
                or any(axis not in "TCZ" for axis in axes[:-2])):
            raise ValueError(f"unsupported plane layout: {axes} {shape}")
        dimensions = dict(zip(axes, shape))
        if dimensions.get("T", 1) != 1:
            raise ValueError("multiple time points require explicit time selection")
        for z, channel in selections:
            if not 0 <= z < dimensions.get("Z", 1):
                raise ValueError("z index is outside this series")
            if not 0 <= channel < dimensions["C"]:
                raise ValueError("channel index is outside this series")
        page_shape = shape[:-2]
        if len(series.pages) != prod(page_shape):
            raise ValueError("series does not store one scalar YX plane per page")
        indices = []
        for z, channel in selections:
            coordinates = {"T": 0, "Z": z, "C": channel}
            index = int(np.ravel_multi_index(
                tuple(coordinates[axis] for axis in axes[:-2]), page_shape
            ))
            page = series.pages[index]
            if tuple(page.shape) != tuple(shape[-2:]):
                raise ValueError("selected page is not a scalar YX image")
            indices.append(index)
        arrays = []
        for index in indices:
            array = series.pages[index].asarray()
            if array.dtype.kind not in "uif" or not np.isfinite(array).all():
                raise ValueError("selected image contains unsupported or non-finite intensities")
            arrays.append(array)
    return arrays, {
        "source_path": str(path), "series_index": series_index, "source_axes": axes,
        "source_shape": list(shape), "page_indices": indices,
        "pair_shape": list(arrays[0].shape), "dtype": str(arrays[0].dtype),
        "raw_values_preserved": True, "pages_decoded": len(indices),
        "registration_status": "unverified",
    }


def load_pair(path: Path, *, z: int, input_channel: int,
              target_channel: int, series_index: int = 0) -> PlanePair:
    """Select explicit channels at one depth and preserve raw values."""
    if input_channel == target_channel:
        raise ValueError("input and target channels must differ")
    arrays, metadata = _load_selected(path, [(z, input_channel), (z, target_channel)], series_index)
    return PlanePair(*arrays, metadata | {"z": z, "input_channel": input_channel,
                                         "target_channel": target_channel})


def load_context(path: Path, *, z: int, input_channel: int,
                 target_channel: int, series_index: int = 0) -> PlanePair:
    """Read only three neighbouring inputs and the centre target, never adjacent targets.

    Input is CYX in z-1,z,z+1 order; target is YX at z. Boundary depths are
    rejected rather than silently replicated, wrapped or dropped.
    """
    if input_channel == target_channel:
        raise ValueError("input and target channels must differ")
    depths = [z - 1, z, z + 1]
    arrays, metadata = _load_selected(path, [(depth, input_channel) for depth in depths]
                                     + [(z, target_channel)], series_index)
    return PlanePair(np.stack(arrays[:3]), arrays[3], metadata | {
        "z": z, "input_channel": input_channel, "target_channel": target_channel,
        "input_z_indices": depths, "input_axes": "CYX", "target_axes": "YX"})
