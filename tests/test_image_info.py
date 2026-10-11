import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np
    import tifffile
except ImportError:
    np = tifffile = None

from virtual_staining.image_info import read_tiff_info


@unittest.skipIf(tifffile is None, "install numpy and tifffile for image metadata tests")
class ImageInfoTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "example.ome.tiff"

    def test_ome_axis_order_and_channel_names_are_preserved(self):
        tifffile.imwrite(self.path, np.zeros((2, 16, 20), dtype=np.uint16),
                         ome=True, photometric="minisblack",
                         metadata={"axes": "CYX", "Channel": {"Name": ["Brightfield", "DNA"]}})
        info = read_tiff_info(self.path)
        self.assertEqual(info["series"][0]["axes"], "CYX")
        self.assertEqual(info["series"][0]["shape"], [2, 16, 20])
        self.assertEqual(info["series"][0]["dtype"], "uint16")
        self.assertEqual([c["Name"] for c in info["ome_images"][0]["channels"]],
                         ["Brightfield", "DNA"])
        self.assertFalse(info["pixels_loaded"])

    def test_plain_tiff_does_not_invent_channel_identities(self):
        tifffile.imwrite(self.path, np.zeros((16, 20), dtype=np.uint8), ome=False)
        info = read_tiff_info(self.path)
        self.assertFalse(info["is_ome"])
        self.assertEqual(info["ome_images"], [])
        self.assertTrue(any("No OME metadata" in w for w in info["warnings"]))

    def test_invalid_tiff_is_rejected(self):
        self.path.write_bytes(b"not an image")
        with self.assertRaises(ValueError):
            read_tiff_info(self.path)
