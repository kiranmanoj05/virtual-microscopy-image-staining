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


def load_pair(path: Path, *, z: int, input_channel: int,
              target_channel: int, series_index: int = 0) -> PlanePair:
    """Select explicit channels from one field/plane and preserve raw values.

    Supported layouts have scalar YX pages and C plus optional Z/T leading
    axes. T must be singleton. Unsupported packed/volumetric layouts fail
    explicitly instead of falling back to loading the entire image.
    """
    if input_channel == target_channel:
        raise ValueError("input and target channels must differ")
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
        if not 0 <= z < dimensions.get("Z", 1):
            raise ValueError("z index is outside this series")
        for channel in (input_channel, target_channel):
            if not 0 <= channel < dimensions["C"]:
                raise ValueError("channel index is outside this series")
        page_shape = shape[:-2]
        if len(series.pages) != prod(page_shape):
            raise ValueError("series does not store one scalar YX plane per page")
        arrays, indices = [], []
        for channel in (input_channel, target_channel):
            coordinates = {"T": 0, "Z": z, "C": channel}
            index = int(np.ravel_multi_index(
                tuple(coordinates[axis] for axis in axes[:-2]), page_shape
            ))
            page = series.pages[index]
            if tuple(page.shape) != tuple(shape[-2:]):
                raise ValueError("selected page is not a scalar YX image")
            array = page.asarray()
            if array.dtype.kind not in "uif" or not np.isfinite(array).all():
                raise ValueError("selected image contains unsupported or non-finite intensities")
            arrays.append(array)
            indices.append(index)
    return PlanePair(arrays[0], arrays[1], {
        "source_path": str(path), "series_index": series_index, "source_axes": axes,
        "source_shape": list(shape), "z": z, "input_channel": input_channel,
        "target_channel": target_channel, "page_indices": indices,
        "pair_shape": list(arrays[0].shape), "dtype": str(arrays[0].dtype),
        "raw_values_preserved": True, "pages_decoded": 2,
        "registration_status": "unverified",
    })
