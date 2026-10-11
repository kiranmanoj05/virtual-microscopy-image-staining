import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np
    import tifffile
except ImportError:
    np = tifffile = None

if tifffile is not None:
    from virtual_staining.paired_planes import load_pair


@unittest.skipIf(tifffile is None, "install numpy and tifffile for plane tests")
class PairedPlaneTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "planes.ome.tiff"

    def write(self, axes="ZCYX", compression=None):
        shape = (3, 2, 8, 12)
        data = np.arange(np.prod(shape), dtype=np.uint16).reshape(shape)
        tifffile.imwrite(self.path, data, ome=True, photometric="minisblack",
                         compression=compression, metadata={"axes": axes})
        return data

    def test_zcyx_preserves_requested_pixels_and_bit_depth(self):
        data = self.write()
        pair = load_pair(self.path, z=1, input_channel=1, target_channel=0)
        np.testing.assert_array_equal(pair.input, data[1, 1])
        np.testing.assert_array_equal(pair.target, data[1, 0])
        self.assertEqual(pair.input.dtype, np.dtype("uint16"))
        self.assertEqual(pair.metadata["page_indices"], [3, 2])

    def test_czyx_does_not_assume_channel_is_second_axis(self):
        data = self.write("CZYX")
        pair = load_pair(self.path, z=1, input_channel=2, target_channel=0)
        np.testing.assert_array_equal(pair.input, data[2, 1])
        np.testing.assert_array_equal(pair.target, data[0, 1])

    def test_lzw_compressed_planes_decode(self):
        try:
            import imagecodecs
        except ImportError:
            self.skipTest("LZW decoding requires imagecodecs")
        data = self.write(compression="lzw")
        pair = load_pair(self.path, z=2, input_channel=0, target_channel=1)
        np.testing.assert_array_equal(pair.input, data[2, 0])
        np.testing.assert_array_equal(pair.target, data[2, 1])

    def test_invalid_indices_fail_instead_of_wrapping(self):
        self.write()
        for z, input_channel, target_channel in [(-1, 0, 1), (3, 0, 1), (0, -1, 1), (0, 0, 2)]:
            with self.subTest(z=z, input_channel=input_channel):
                with self.assertRaises(ValueError):
                    load_pair(self.path, z=z, input_channel=input_channel, target_channel=target_channel)

    def test_same_channel_cannot_be_input_and_target(self):
        self.write()
        with self.assertRaises(ValueError):
            load_pair(self.path, z=0, input_channel=0, target_channel=0)

    def test_multiple_timepoints_are_not_silently_collapsed(self):
        tifffile.imwrite(self.path, np.zeros((2, 2, 8, 12), dtype=np.uint16),
                         ome=True, photometric="minisblack", metadata={"axes": "TCYX"})
        with self.assertRaisesRegex(ValueError, "time"):
            load_pair(self.path, z=0, input_channel=0, target_channel=1)
