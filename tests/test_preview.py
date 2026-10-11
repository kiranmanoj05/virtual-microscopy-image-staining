import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

available = all(importlib.util.find_spec(name) for name in ("numpy", "tifffile", "matplotlib"))
if available:
    import numpy as np
    import tifffile
    from virtual_staining.preview import preview_pair


@unittest.skipUnless(available, "install preview dependencies")
class PreviewTests(unittest.TestCase):
    def test_measurements_export_with_a_shared_scale_across_depths(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "example.ome.tiff"
            data = np.zeros((2, 2, 16, 16), dtype=np.uint16)
            data[0, 0] = 100
            data[1, 0] = 200
            data[0, 1] = 1000
            data[1, 1] = 2000
            tifffile.imwrite(path, data, ome=True, photometric="minisblack", metadata={"axes": "ZCYX"})
            output = Path(temp) / "preview"
            report = preview_pair(path, z_indices=[0, 1], input_channel=0,
                                  target_channel=1, out_dir=output)
            self.assertEqual(report["display_scaling"]["input_range"], [100, 200])
            self.assertEqual(report["display_scaling"]["target_range"], [1000, 2000])
            self.assertEqual(report["pages_decoded"], 4)
            self.assertFalse(report["predictions_generated"])
            self.assertEqual((output / "preview.png").read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            self.assertEqual(json.loads((output / "preview.json").read_text()), report)

    def test_excessive_preview_requests_fail_before_reading(self):
        with self.assertRaises(ValueError):
            preview_pair(Path("missing.tiff"), z_indices=list(range(9)),
                         input_channel=0, target_channel=1, out_dir=Path("unused"))
