import json
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import tifffile

from virtual_staining.volume_data import prepare_volumes, load_volumes, standardise_volume


class VolumeDataTests(unittest.TestCase):
    def fixture(self):
        from test_pilot_audit import PilotAuditTests
        f = PilotAuditTests()
        f.setUp()
        self.addCleanup(f.temp.cleanup)
        for field in f.plan["fields"]:
            a = np.arange(16 * 3 * 32 * 32, dtype=np.uint16).reshape(16, 3, 32, 32)
            tifffile.imwrite(f.file_path(field), a, photometric="minisblack",
                             metadata={"axes": "ZCYX", "PhysicalSizeX": 0.29,
                                       "PhysicalSizeY": 0.29, "PhysicalSizeZ": 0.29,
                                       "Channel": {"Fluor": ["EGFP", "Hoechst", "TL Brightfield"]}})
            field["size_bytes"] = f.file_path(field).stat().st_size
        f.freeze()
        receipts = {"pilot_plan_sha256": f.plan["plan_sha256"], "records": [
            json.loads((f.raw / f"allen_fov_{field['fov_id']}.json").read_text())
            for field in f.plan["fields"]]}
        f.receipts = f.root / "original-receipts.json"
        f.receipts.write_text(json.dumps(receipts))
        return f

    def test_zscore_removes_offset_gain_and_rejects_constant_nonfinite(self):
        a = np.arange(64).reshape(4, 4, 4)
        x, stats = standardise_volume(a)
        y, _ = standardise_volume(3 * a + 90)
        np.testing.assert_allclose(x, y)
        self.assertAlmostEqual(float(x.mean()), 0)
        self.assertAlmostEqual(float(x.std()), 1)
        self.assertEqual(stats["mean"], float(a.mean()))
        for bad in (np.ones((4, 4, 4)), np.full((4, 4, 4), np.nan)):
            with self.assertRaises(ValueError):
                standardise_volume(bad)

    def test_real_volume_preparation_ignores_absent_test_and_preserves_all_depths(self):
        f = self.fixture()
        test = next(x for x in f.plan["fields"] if x["split"] == "test")
        f.file_path(test).unlink()
        report = prepare_volumes(f.plan_path, f.receipts, f.raw, f.root / "volumes")
        self.assertFalse(report["test_pixels_loaded"])
        metadata, _, fields = load_volumes(f.root / "volumes/volumes.json")
        self.assertEqual(len(fields), 2)
        self.assertEqual(fields[0][1].shape, (16, 32, 32))
        self.assertEqual(fields[0][1].dtype, np.float32)
        self.assertEqual(len(metadata["fields"][0]["page_indices"]), 32)
        self.assertIn("no raw-intensity calibration", metadata["target_units"])
        with self.assertRaisesRegex(ValueError, "already exists"):
            prepare_volumes(f.plan_path, f.receipts, f.raw, f.root / "volumes")

    def test_tampered_raw_fails_before_decoding_and_cleans_staging(self):
        f = self.fixture()
        field = next(x for x in f.plan["fields"] if x["split"] == "train")
        with f.file_path(field).open("ab") as s:
            s.write(b"changed")
        from virtual_staining.paired_planes import _load_selected
        with patch("virtual_staining.volume_data._load_selected", wraps=_load_selected) as decoder:
            with self.assertRaisesRegex(ValueError, "original receipt"):
                prepare_volumes(f.plan_path, f.receipts, f.raw, f.root / "volumes")
        self.assertTrue(all(call.args[0] != f.file_path(field) for call in decoder.call_args_list))
        # Validation might be listed first, but changed training bytes never decode.
        self.assertFalse((f.root / "volumes").exists())
        self.assertEqual(list(f.root.glob(".volumes-*")), [])

    def test_changed_prepared_array_fails(self):
        f = self.fixture()
        prepare_volumes(f.plan_path, f.receipts, f.raw, f.root / "volumes")
        with (f.root / "volumes/000_input.npy").open("ab") as s:
            s.write(b"changed")
        with self.assertRaisesRegex(ValueError, "checksum"):
            load_volumes(f.root / "volumes/volumes.json")

    def test_unknown_units_fail_instead_of_guessing_physical_scale(self):
        f = self.fixture()
        from virtual_staining.image_info import read_tiff_info
        def info(path):
            r = read_tiff_info(path)
            r["ome_images"][0]["pixels"]["PhysicalSizeXUnit"] = "nm"
            return r
        with patch("virtual_staining.volume_data.read_tiff_info", side_effect=info):
            with self.assertRaisesRegex(ValueError, "micrometres"):
                prepare_volumes(f.plan_path, f.receipts, f.raw, f.root / "volumes")
