import hashlib
import json
import unittest

try:
    import numpy as np
    import torch
    from virtual_staining.data_diagnostics import diagnose_data, channel_summary, signal_summary
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "install the ml extra for data diagnostics")
class DataDiagnosticTests(unittest.TestCase):
    def fixture(self):
        from test_trainer import TrainerTests
        case = TrainerTests()
        f = case.fixture()
        self.addCleanup(f.temp.cleanup)
        return f

    def test_raw_shift_remains_visible_after_model_clipping(self):
        result = channel_summary(np.array([[0, 10, 20, 40]], dtype=np.uint16),
                                 {"low": 10, "high": 20, "clip": True})
        self.assertEqual(result["model_values"]["max"], 1)
        self.assertEqual(result["scaled_before_clipping"]["max"], 3)
        self.assertEqual(result["raw"]["max"], 40)
        self.assertEqual(result["fraction_above_training_high"], 0.25)
        self.assertEqual(result["fraction_below_training_low"], 0.25)

    def test_bright_error_fraction_and_empty_signal_are_explicit(self):
        result = signal_summary(np.array([[0, 0, 0, 2]], dtype=np.float32), 0)
        self.assertEqual(result["bright_pixel_fraction"], 0.25)
        self.assertEqual(result["bright_fraction_of_constant_absolute_error"], 1)
        result = signal_summary(np.zeros((8, 8), dtype=np.float32), 0)
        self.assertIsNone(result["bright_constant_mae"])
        self.assertIsNone(result["bright_fraction_of_constant_absolute_error"])

    def test_checkpoint_free_report_and_validation_cannot_fit_baseline(self):
        f = self.fixture()
        path = f.root / "prepared/dataset.json"
        first = diagnose_data(path, f.root / "first", patch_size=8)
        metadata = json.loads(path.read_text())
        validation = next(r for r in metadata["fields"] if r["split"] == "validation")
        target_path = path.parent / validation["target_file"]
        np.save(target_path, np.full((8, 8), 60000, dtype=np.uint16))
        validation["target_sha256"] = hashlib.sha256(target_path.read_bytes()).hexdigest()
        metadata.pop("prepared_metadata_sha256")
        metadata["prepared_metadata_sha256"] = hashlib.sha256(
            json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        path.write_text(json.dumps(metadata))
        second = diagnose_data(path, f.root / "second", patch_size=8)
        self.assertEqual(first["constant_value_scaled"], second["constant_value_scaled"])
        self.assertEqual(first["scaling"], second["scaling"])
        report = json.loads((f.root / "second/data-diagnostics.json").read_text())
        self.assertFalse(report["test_pixels_loaded"])
        self.assertEqual({r["split"] for r in report["fields"]}, {"train", "validation"})
        self.assertEqual(len(report["fields"][0]["fixed_patches"]), 1)
        with self.assertRaisesRegex(ValueError, "already exists"):
            diagnose_data(path, f.root / "second", patch_size=8)
        with self.assertRaisesRegex(ValueError, "does not fit"):
            diagnose_data(path, f.root / "too-big", patch_size=16)
        self.assertFalse((f.root / "too-big").exists())

    def test_changed_pixels_rejected_before_creating_report(self):
        f = self.fixture()
        path = f.root / "prepared/dataset.json"
        metadata = json.loads(path.read_text())
        with (path.parent / metadata["fields"][0]["input_file"]).open("ab") as stream:
            stream.write(b"changed")
        with self.assertRaisesRegex(ValueError, "checksum"):
            diagnose_data(path, f.root / "invalid", patch_size=8)
        self.assertFalse((f.root / "invalid").exists())

    def test_test_entry_rejected_before_pixel_access(self):
        f = self.fixture()
        path = f.root / "prepared/dataset.json"
        metadata = json.loads(path.read_text())
        metadata["fields"][0]["split"] = "test"
        metadata["fields"][0]["input_file"] = "does-not-exist.npy"
        metadata.pop("prepared_metadata_sha256")
        metadata["prepared_metadata_sha256"] = hashlib.sha256(
            json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        path.write_text(json.dumps(metadata))
        with self.assertRaisesRegex(ValueError, "development fields only"):
            diagnose_data(path, f.root / "invalid", patch_size=8)


if __name__ == "__main__":
    unittest.main()
