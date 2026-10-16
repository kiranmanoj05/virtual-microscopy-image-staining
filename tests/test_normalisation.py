import hashlib
import json
import unittest

try:
    import numpy as np
    import torch
    from virtual_staining.training_data import input_scaling, scale_plane
    from virtual_staining.trainer import load_development_data, train_pilot, sample_batch
    from virtual_staining.diagnostics import diagnose_training
    from virtual_staining.preprocessing_comparison import compare_normalisation
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "install the ml extra for normalisation checks")
class NormalisationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def fixture(self):
        from test_trainer import TrainerTests
        f = TrainerTests().fixture()
        self.addCleanup(f.temp.cleanup)
        return f

    def train(self, f, name, policy, steps=2, resume=None):
        return train_pilot(f.root / "prepared/dataset.json", f.root / name, steps=steps,
                           batch_size=1, patch_size=8, width=2, seed=17, validate_every=2,
                           device_name="cpu", objective="mse", input_normalisation=policy,
                           resume_path=resume)

    def test_per_plane_is_input_affine_invariant_and_constant_planes_fail(self):
        image = np.arange(256, dtype=np.float32).reshape(16, 16)
        changed = image * 2 + 1000
        reference = {"low": 0, "high": 255, "clip": True}
        a = scale_plane(image, input_scaling(image, reference, "per-plane"))
        b = scale_plane(changed, input_scaling(changed, reference, "per-plane"))
        np.testing.assert_allclose(a, b, atol=1e-6)
        self.assertFalse(np.allclose(a, scale_plane(changed, reference)))
        with self.assertRaisesRegex(ValueError, "positive percentile range"):
            input_scaling(np.ones((8, 8)), reference, "per-plane")
        with self.assertRaisesRegex(ValueError, "normalisation"):
            input_scaling(image, reference, "invalid")

    def test_loader_preserves_targets_and_fits_only_each_input(self):
        f = self.fixture()
        path = f.root / "prepared/dataset.json"
        _, identity, global_data = load_development_data(path)
        _, other_identity, planes = load_development_data(path, input_normalisation="per-plane")
        self.assertEqual(identity, other_identity)
        for a, b in zip(global_data, planes):
            np.testing.assert_array_equal(a[2], b[2])
        before = planes[0][0]["effective_input_scaling"]
        metadata = json.loads(path.read_text())
        field = metadata["fields"][0]
        target_path = path.parent / field["target_file"]
        np.save(target_path, np.full((8, 8), 60000, dtype=np.uint16))
        field["target_sha256"] = hashlib.sha256(target_path.read_bytes()).hexdigest()
        metadata.pop("prepared_metadata_sha256")
        metadata["prepared_metadata_sha256"] = hashlib.sha256(
            json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        path.write_text(json.dumps(metadata))
        _, _, changed = load_development_data(path, input_normalisation="per-plane")
        self.assertEqual(before, changed[0][0]["effective_input_scaling"])
        np.testing.assert_array_equal(planes[0][1], changed[0][1])

    def test_trace_preserves_random_choices_when_input_values_change(self):
        image = np.arange(256, dtype=np.float32).reshape(16, 16)
        a, b = hashlib.sha256(), hashlib.sha256()
        _, left_target = sample_batch([({"source_id": "a"}, image, image)],
                                     np.random.default_rng(17), 8, 10, trace_digest=a)
        _, right_target = sample_batch([({"source_id": "a"}, image * 2, image)],
                                      np.random.default_rng(17), 8, 10, trace_digest=b)
        self.assertEqual(a.hexdigest(), b.hexdigest())
        torch.testing.assert_close(left_target, right_target)

    def test_fresh_comparison_and_diagnostics_restore_saved_policy(self):
        f = self.fixture()
        global_run = self.train(f, "global", "global")
        plane_run = self.train(f, "plane", "per-plane")
        self.assertEqual(global_run["target_batches_sha256"], plane_run["target_batches_sha256"])
        self.assertEqual(global_run["sampler_trace_sha256"], plane_run["sampler_trace_sha256"])
        result = compare_normalisation(f.root / "global", f.root / "plane", f.root / "comparison")
        self.assertEqual(result["objective"], "mse")
        self.assertFalse(result["test_pixels_loaded"])
        diagnose_training(f.root / "prepared/dataset.json", f.root / "plane", f.root / "diagnostics")
        report = json.loads((f.root / "diagnostics/diagnostics.json").read_text())
        self.assertEqual(report["input_normalisation"], "per-plane")
        path = f.root / "plane/training.json"
        run = json.loads(path.read_text())
        run["config"]["input_normalisation"] = "global"
        path.write_text(json.dumps(run))
        with self.assertRaisesRegex(ValueError, "preprocessing fingerprint"):
            diagnose_training(f.root / "prepared/dataset.json", f.root / "plane", f.root / "rejected")

    def test_policy_locked_resume_and_uninterrupted_weights_match(self):
        f = self.fixture()
        self.train(f, "first", "per-plane")
        with self.assertRaisesRegex(ValueError, "input_normalisation"):
            self.train(f, "wrong", "global", 4, f.root / "first/last-model.pt")
        self.assertFalse((f.root / "wrong").exists())
        self.train(f, "resumed", "per-plane", 4, f.root / "first/last-model.pt")
        self.train(f, "full", "per-plane", 4)
        a = torch.load(f.root / "resumed/last-model.pt", weights_only=True)
        b = torch.load(f.root / "full/last-model.pt", weights_only=True)
        for key in a["model_state"]:
            self.assertTrue(torch.equal(a["model_state"][key], b["model_state"][key]))

    def test_comparison_rejects_changed_targets_or_other_controls(self):
        f = self.fixture()
        self.train(f, "global", "global")
        self.train(f, "plane", "per-plane")
        path = f.root / "plane/training.json"
        original = json.loads(path.read_text())
        for key in ("target_batches_sha256", "sampler_trace_sha256"):
            changed = json.loads(json.dumps(original))
            changed[key] = "f" * 64
            path.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, "sampling or targets"):
                compare_normalisation(f.root / "global", f.root / "plane", f.root / "rejected")
        changed = json.loads(json.dumps(original))
        changed["config"]["objective"] = "l1"
        path.write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError, "normalisation-only"):
            compare_normalisation(f.root / "global", f.root / "plane", f.root / "rejected")
        self.assertFalse((f.root / "rejected").exists())


if __name__ == "__main__":
    unittest.main()
