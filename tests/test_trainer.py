import json
import unittest

try:
    import torch
    import numpy as np
    from virtual_staining.trainer import train_pilot, sample_batch, image_scores, summarise_scores
    from virtual_staining.training_data import prepare_training_data
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "install the ml extra for trainer checks")
class TrainerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def fixture(self):
        from test_pilot_audit import PilotAuditTests
        from virtual_staining.pilot_audit import audit_pilot
        f = PilotAuditTests()
        f.setUp()
        self.addCleanup(f.temp.cleanup)
        audit_pilot(f.plan_path, f.raw, f.root / "audit")
        prepare_training_data(f.root / "audit/audit.json", f.root / "prepared")
        return f

    def run_training(self, f, name, steps, resume=None):
        return train_pilot(f.root / "prepared/dataset.json", f.root / name, steps=steps,
                           batch_size=1, patch_size=8, width=2, seed=17,
                           validate_every=2, device_name="cpu", resume_path=resume)

    def test_paired_augmentations_preserve_correspondence(self):
        image = np.arange(256, dtype=np.float32).reshape(16, 16)
        inputs, targets = sample_batch([({}, image, image + 7)], np.random.default_rng(17), 8, 5)
        torch.testing.assert_close(targets - inputs, torch.full_like(inputs, 7))

    def test_whole_image_metrics_and_group_balanced_means(self):
        image = np.arange(64, dtype=np.float32).reshape(8, 8) / 63
        scores = image_scores(image, image)
        self.assertEqual(scores["mae"], 0)
        self.assertAlmostEqual(scores["pearson"], 1)
        self.assertAlmostEqual(scores["ssim"], 1)
        records = [{"group_id": "a", "mae": 0, "pearson": None, "ssim": 1}] * 5
        records += [{"group_id": "b", "mae": 1, "pearson": 0.5, "ssim": 0}]
        result = summarise_scores(records)
        self.assertEqual(result["mae"], 0.5)
        self.assertEqual(result["undefined_pearson_fields"], 5)

    def test_end_to_end_validation_checkpoints_and_resume_match_uninterrupted_updates(self):
        f = self.fixture()
        first = self.run_training(f, "first", 2)
        resumed = self.run_training(f, "resumed", 4, f.root / "first/last-model.pt")
        full = self.run_training(f, "full", 4)
        a = torch.load(f.root / "resumed/last-model.pt", weights_only=True)
        b = torch.load(f.root / "full/last-model.pt", weights_only=True)
        for key in a["model_state"]:
            self.assertTrue(torch.equal(a["model_state"][key], b["model_state"][key]))
        self.assertEqual(resumed["steps_this_run"], 2)
        self.assertFalse(first["test_pixels_loaded"])
        self.assertEqual(first["validation_fields"], 1)
        self.assertTrue((f.root / "first/validation-preview.png").exists())
        self.assertIsNone(first["constant_baseline"]["pearson"])
        self.assertEqual(full["config"]["ssim_data_range_scaled"], 1)

    def test_modified_checkpoint_cannot_resume(self):
        f = self.fixture()
        self.run_training(f, "first", 2)
        checkpoint = f.root / "first/last-model.pt"
        with checkpoint.open("ab") as stream:
            stream.write(b"changed")
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            self.run_training(f, "resumed", 4, checkpoint)


if __name__ == "__main__":
    unittest.main()
