import json
import unittest
from unittest.mock import patch

try:
    import numpy as np
    import torch
    from virtual_staining.model import UNet
    from virtual_staining.overfit import overfit_patch
    from virtual_staining.training_data import fit_scaling, prepare_training_data, scale_plane
    from virtual_staining.paired_planes import load_pair
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "install the ml extra for training checks")
class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def fixture(self, validation_offset=0):
        from test_pilot_audit import PilotAuditTests
        from virtual_staining.pilot_audit import audit_pilot
        fixture = PilotAuditTests()
        fixture.setUp()
        self.addCleanup(fixture.temp.cleanup)
        if validation_offset:
            import tifffile
            field = next(f for f in fixture.plan["fields"] if f["split"] == "validation")
            pixels = np.arange(4 * 3 * 8 * 8, dtype=np.uint16).reshape(4, 3, 8, 8) + validation_offset
            tifffile.imwrite(fixture.file_path(field), pixels, photometric="minisblack",
                             metadata={"axes": "ZCYX", "Channel": {"Fluor": ["EGFP", "Hoechst 33342", "TL Brightfield"]}})
            field["size_bytes"] = fixture.file_path(field).stat().st_size
            fixture.freeze()
        audit_pilot(fixture.plan_path, fixture.raw, fixture.root / "audit")
        return fixture

    def test_model_preserves_odd_image_size_and_has_finite_gradients(self):
        torch.manual_seed(17)
        model = UNet(width=2)
        image = torch.rand(2, 1, 31, 35)
        prediction = model(image)
        self.assertEqual(prediction.shape, image.shape)
        (prediction - image).abs().mean().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_invalid_model_shape_fails_explicitly(self):
        model = UNet(width=2)
        for shape in [(1, 2, 16, 16), (1, 1, 7, 16), (16, 16)]:
            with self.assertRaises(ValueError):
                model(torch.zeros(shape))

    def test_target_extremes_are_preserved_while_input_is_clipped(self):
        source = np.arange(100, dtype=np.uint16).reshape(10, 10)
        scaling = fit_scaling([(source, source)])
        extreme = np.array([[-10, 1000]], dtype=np.float32)
        target = scale_plane(extreme, scaling["target"])
        inputs = scale_plane(extreme, scaling["input"])
        self.assertLess(target.min(), 0)
        self.assertGreater(target.max(), 1)
        np.testing.assert_array_equal(inputs, [[0, 1]])

    def test_preparation_keeps_raw_planes_and_test_is_excluded(self):
        fixture = self.fixture(validation_offset=10000)
        with patch("virtual_staining.training_data.load_pair", wraps=load_pair) as loader:
            report = prepare_training_data(fixture.root / "audit/audit.json", fixture.root / "prepared")
        self.assertEqual(loader.call_count, 2)
        self.assertEqual(report["train_fields"], 1)
        self.assertEqual(report["validation_fields"], 1)
        metadata = json.loads((fixture.root / "prepared/dataset.json").read_text())
        train = next(f for f in metadata["fields"] if f["split"] == "train")
        target = np.load(fixture.root / "prepared" / train["target_file"])
        self.assertEqual(target.dtype, np.uint16)
        self.assertEqual(train["z"], 2)
        self.assertEqual(len(metadata["fields"]), 2)
        self.assertEqual(report["scaling"]["target"]["low"], float(np.percentile(target, 1)))

    def test_cross_split_group_is_rejected(self):
        fixture = self.fixture()
        path = fixture.root / "audit/audit.json"
        audit = json.loads(path.read_text())
        audit["fields"][1]["group_id"] = audit["fields"][0]["group_id"]
        path.write_text(json.dumps(audit))
        with self.assertRaisesRegex(ValueError, "crosses splits"):
            prepare_training_data(path, fixture.root / "prepared")

    def test_one_patch_learning_decreases_loss_and_records_provenance(self):
        fixture = self.fixture()
        prepare_training_data(fixture.root / "audit/audit.json", fixture.root / "prepared")
        result = overfit_patch(fixture.root / "prepared/dataset.json", fixture.root / "overfit",
                              steps=60, patch_size=8, width=2, device_name="cpu")
        self.assertTrue(result["loss_decreased"])
        self.assertEqual(len(result["checkpoint_sha256"]), 64)
        self.assertFalse(result["test_pixels_loaded"])
        self.assertTrue((fixture.root / "overfit/overfit.json").exists())

    def test_changed_prepared_pixels_fail_before_training(self):
        fixture = self.fixture()
        prepare_training_data(fixture.root / "audit/audit.json", fixture.root / "prepared")
        path = fixture.root / "prepared/dataset.json"
        metadata = json.loads(path.read_text())
        train = next(f for f in metadata["fields"] if f["split"] == "train")
        (path.parent / train["input_file"]).write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "checksum"):
            overfit_patch(path, fixture.root / "overfit", patch_size=8, width=2, device_name="cpu")


if __name__ == "__main__":
    unittest.main()
