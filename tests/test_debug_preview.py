import unittest

try:
    import torch
    import numpy as np
    from virtual_staining.debug_preview import patch_scores, preview_overfit
    from virtual_staining.training_data import prepare_training_data
    from virtual_staining.overfit import overfit_patch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "install the ml extra for prediction previews")
class DebugPreviewTests(unittest.TestCase):
    def test_constant_pearson_is_undefined_and_identical_arrays_score_perfectly(self):
        target = np.arange(16, dtype=np.float32).reshape(4, 4)
        self.assertIsNone(patch_scores(np.ones_like(target), target)["pearson"])
        self.assertEqual(patch_scores(target, target)["mae"], 0)
        self.assertAlmostEqual(patch_scores(target, target)["pearson"], 1)

    def test_metrics_reject_nonfinite_predictions(self):
        with self.assertRaises(ValueError):
            patch_scores(np.array([[np.nan]]), np.array([[0]]))

    def test_saved_model_previews_the_bound_patch_and_rejects_changed_checkpoint(self):
        from test_pilot_audit import PilotAuditTests
        from virtual_staining.pilot_audit import audit_pilot
        fixture = PilotAuditTests()
        fixture.setUp()
        self.addCleanup(fixture.temp.cleanup)
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, threads)
        audit_pilot(fixture.plan_path, fixture.raw, fixture.root / "audit")
        prepare_training_data(fixture.root / "audit/audit.json", fixture.root / "prepared")
        dataset = fixture.root / "prepared/dataset.json"
        overfit_patch(dataset, fixture.root / "debug", steps=2, patch_size=8, width=2, device_name="cpu")
        checkpoint = fixture.root / "debug/debug-model.pt"
        result = preview_overfit(dataset, checkpoint, fixture.root / "preview")
        self.assertFalse(result["test_pixels_loaded"])
        self.assertIsNone(result["constant_scores"]["pearson"])
        self.assertTrue((fixture.root / "preview/preview.png").exists())
        with checkpoint.open("ab") as stream:
            stream.write(b"modified")
        with self.assertRaisesRegex(ValueError, "checksum"):
            preview_overfit(dataset, checkpoint, fixture.root / "other-preview")


if __name__ == "__main__":
    unittest.main()
