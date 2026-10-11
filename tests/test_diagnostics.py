import unittest

try:
    import torch
    import numpy as np
    from virtual_staining.diagnostics import crop_locations, intensity_summary, diagnose_training
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "install the ml extra for diagnostics")
class DiagnosticTests(unittest.TestCase):
    def test_fixed_locations_include_center_and_corners_without_duplicates(self):
        self.assertEqual(len(crop_locations((24, 32), 8)), 5)
        self.assertEqual(crop_locations((8, 8), 8), [(0, 0)])
        with self.assertRaises(ValueError):
            crop_locations((8, 8), 16)

    def test_intensity_summary_reports_clipped_endpoints(self):
        result = intensity_summary(np.array([[0, 0.5, 1, 2]], dtype=np.float32))
        self.assertEqual(result["fraction_le_zero"], 0.25)
        self.assertEqual(result["fraction_ge_one"], 0.5)

    def test_diagnostics_compare_both_checkpoints_without_test_sources(self):
        from test_trainer import TrainerTests
        fixture_case = TrainerTests()
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, previous_threads)
        fixture = fixture_case.fixture()
        self.addCleanup(fixture.temp.cleanup)
        fixture_case.run_training(fixture, "training", 2)
        result = diagnose_training(fixture.root / "prepared/dataset.json", fixture.root / "training",
                                   fixture.root / "diagnostics")
        self.assertFalse(result["test_pixels_loaded"])
        self.assertEqual(set(result["checkpoints"]), {"best", "last"})
        import json
        full = json.loads((fixture.root / "diagnostics/diagnostics.json").read_text())
        for checkpoint in full["checkpoints"].values():
            self.assertEqual({f["split"] for f in checkpoint["fields"]}, {"train", "validation"})
            self.assertTrue(all(len(f["patch_context"]) == 1 for f in checkpoint["fields"]))
        self.assertTrue((fixture.root / "diagnostics/best-train/training-preview.png").exists())


if __name__ == "__main__":
    unittest.main()
