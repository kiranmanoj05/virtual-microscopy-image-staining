"""Inspect TIFF metadata without loading image pixels or guessing channels."""

import xml.etree.ElementTree as ET
from pathlib import Path


def read_tiff_info(path: Path) -> dict:
    import tifffile

    report = {"path": str(path), "size_bytes": path.stat().st_size,
              "pixels_loaded": False, "series": [], "ome_images": [], "warnings": []}
    with tifffile.TiffFile(path) as image:
        report["is_ome"] = image.is_ome
        for index, series in enumerate(image.series):
            report["series"].append({"index": index, "axes": series.axes,
                                     "shape": list(series.shape), "dtype": str(series.dtype)})
        if image.ome_metadata:
            try:
                root = ET.fromstring(image.ome_metadata)
            except ET.ParseError as error:
                report["warnings"].append(f"OME XML could not be parsed: {error}")
                return report
            # Match local tag names to tolerate OME namespace revisions.
            for node in root:
                if node.tag.rsplit("}", 1)[-1] != "Image":
                    continue
                entry = {"id": node.get("ID"), "name": node.get("Name"),
                         "pixels": {}, "channels": []}
                for child in node:
                    if child.tag.rsplit("}", 1)[-1] != "Pixels":
                        continue
                    entry["pixels"] = dict(child.attrib)
                    for channel in child:
                        if channel.tag.rsplit("}", 1)[-1] == "Channel":
                            entry["channels"].append({"index": len(entry["channels"]),
                                                       **dict(channel.attrib)})
                report["ome_images"].append(entry)
        else:
            report["warnings"].append("No OME metadata: channel identities require another source.")
    report["warnings"].append(
        "Channel metadata is descriptive; pairing, registration and split independence are not verified."
    )
    return report
