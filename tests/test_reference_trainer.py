import json
import unittest

import numpy as np
import torch

from virtual_staining.reference_trainer import make_model, predict_volume, train_reference, volume_batch
from virtual_staining.volume_data import prepare_volumes


class ReferenceTrainerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_tiled_inference_covers_every_voxel_without_target_or_seams_for_pointwise_model(self):
        model = torch.nn.Conv3d(1, 1, 1)
        with torch.no_grad():
            model.weight.fill_(2)
            model.bias.fill_(3)
        a = np.random.default_rng(17).normal(size=(19, 35, 43)).astype(np.float32)
        predicted = predict_volume(model, a, core=(8, 16, 16), halo=3, divisor=8)
        np.testing.assert_allclose(predicted, 2 * a + 3, rtol=1e-6, atol=1e-6)
        self.assertEqual(predicted.shape, a.shape)
        self.assertFalse(model.training)

    def test_sampling_preserves_3d_correspondence_and_is_reproducible(self):
        a = np.arange(32 ** 3, dtype=np.float32).reshape(32, 32, 32)
        fields = [({"source_id": "train"}, a, a + 7)]
        x, y, positions = volume_batch(fields, np.random.default_rng(17), (8, 16, 16), 3)
        torch.testing.assert_close(y - x, torch.full_like(x, 7))
        _, _, repeated = volume_batch(fields, np.random.default_rng(17), (8, 16, 16), 3)
        self.assertEqual(positions, repeated)
        self.assertEqual(tuple(x.shape), (3, 1, 8, 16, 16))

    def test_reference_architecture_has_finite_gradients_and_expected_capacity(self):
        torch.manual_seed(17)
        model = make_model(4, 4)
        x = torch.randn(2, 1, 16, 16, 16)
        prediction = model(x)
        self.assertEqual(tuple(prediction.shape), tuple(x.shape))
        prediction.square().mean().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        self.assertEqual(sum(p.numel() for p in model.parameters()), 365233)

    def fixture(self):
        from test_volume_data import VolumeDataTests
        helper = VolumeDataTests()
        f = helper.fixture()
        self.addCleanup(f.temp.cleanup)
        prepare_volumes(f.plan_path, f.receipts, f.raw, f.root / "volumes")
        return f

    def test_end_to_end_saves_complete_validation_and_reloadable_checkpoint(self):
        f = self.fixture()
        result = train_reference(f.root / "volumes/volumes.json", f.root / "run", steps=2,
                                 width=2, depth=2, patch_shape=(8, 16, 16), validate_every=2)
        report = json.loads((f.root / "run/training.json").read_text())
        self.assertFalse(report["test_pixels_loaded"])
        self.assertFalse(report["config"]["architecture_matches_reference"])
        self.assertEqual(result["best_validation"]["fields"], 1)
        checkpoint = torch.load(f.root / "run/best-model.pt", weights_only=True)
        model = make_model(2, 2)
        model.load_state_dict(checkpoint["model_state"], strict=True)
        self.assertTrue((f.root / "run/validation-0.png").is_file())
        self.assertTrue((f.root / "run/validation-0-prediction.npy").is_file())
        with self.assertRaisesRegex(ValueError, "already exists"):
            train_reference(f.root / "volumes/volumes.json", f.root / "run")

    def test_invalid_patch_geometry_fails_without_creating_output(self):
        f = self.fixture()
        with self.assertRaisesRegex(ValueError, "divisible"):
            train_reference(f.root / "volumes/volumes.json", f.root / "bad", patch_shape=(17, 64, 64))
        self.assertFalse((f.root / "bad").exists())
