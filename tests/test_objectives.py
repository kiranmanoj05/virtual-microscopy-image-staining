import hashlib
import json
import unittest

try:
    import torch
    from virtual_staining.trainer import train_pilot, regression_loss, model_state_digest
    from virtual_staining.run_comparison import compare_objectives
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "install the ml extra for objective checks")
class ObjectiveTests(unittest.TestCase):
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

    def run_training(self, f, name, objective, steps=2, resume=None):
        return train_pilot(f.root / "prepared/dataset.json", f.root / name, steps=steps,
                           batch_size=1, patch_size=8, width=2, seed=17, validate_every=2,
                           device_name="cpu", objective=objective, resume_path=resume)

    def test_objective_values_and_gradients(self):
        prediction = torch.tensor([1.0, 3.0], requires_grad=True)
        target = torch.zeros(2)
        loss = regression_loss(prediction, target, "l1")
        self.assertEqual(loss.item(), 2)
        loss.backward()
        torch.testing.assert_close(prediction.grad, torch.tensor([0.5, 0.5]))
        prediction.grad = None
        loss = regression_loss(prediction, target, "mse")
        self.assertEqual(loss.item(), 5)
        loss.backward()
        torch.testing.assert_close(prediction.grad, torch.tensor([1.0, 3.0]))
        with self.assertRaisesRegex(ValueError, "objective"):
            regression_loss(prediction, target, "invalid")

    def test_fresh_runs_have_matching_initialisation_samples_and_validation_controls(self):
        f = self.fixture()
        left = self.run_training(f, "l1", "l1")
        right = self.run_training(f, "mse", "mse")
        self.assertEqual(left["config"]["initial_model_state_sha256"],
                         right["config"]["initial_model_state_sha256"])
        self.assertEqual(left["sampled_batches_sha256"], right["sampled_batches_sha256"])
        result = compare_objectives(f.root / "l1", f.root / "mse", f.root / "comparison")
        self.assertEqual(result["total_steps_each"], 2)
        self.assertFalse(result["test_pixels_loaded"])
        self.assertEqual(left["constant_baseline"], right["constant_baseline"])
        self.assertEqual(left["untrained_model"], right["untrained_model"])
        with self.assertRaisesRegex(ValueError, "already exists"):
            compare_objectives(f.root / "l1", f.root / "mse", f.root / "comparison")

    def test_mse_resume_matches_uninterrupted_weights_and_cannot_switch_objective(self):
        f = self.fixture()
        self.run_training(f, "first", "mse")
        with self.assertRaisesRegex(ValueError, "objective"):
            self.run_training(f, "invalid", "l1", 4, f.root / "first/last-model.pt")
        self.assertFalse((f.root / "invalid").exists())
        self.run_training(f, "resumed", "mse", 4, f.root / "first/last-model.pt")
        self.run_training(f, "full", "mse", 4)
        a = torch.load(f.root / "resumed/last-model.pt", weights_only=True)
        b = torch.load(f.root / "full/last-model.pt", weights_only=True)
        for key in a["model_state"]:
            self.assertTrue(torch.equal(a["model_state"][key], b["model_state"][key]))

    def test_legacy_missing_objective_is_l1_only(self):
        f = self.fixture()
        self.run_training(f, "legacy", "l1")
        checkpoint = f.root / "legacy/last-model.pt"
        saved = torch.load(checkpoint, weights_only=True)
        del saved["config"]["objective"]
        torch.save(saved, checkpoint)
        report_path = f.root / "legacy/training.json"
        report = json.loads(report_path.read_text())
        report["last_checkpoint_sha256"] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        report_path.write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError, "objective"):
            self.run_training(f, "wrong", "mse", 4, checkpoint)
        self.run_training(f, "valid", "l1", 4, checkpoint)

    def test_objective_comparison_rejects_mismatched_controls_sampling_and_warm_start(self):
        f = self.fixture()
        self.run_training(f, "l1", "l1")
        self.run_training(f, "mse", "mse")
        path = f.root / "mse/training.json"
        original = json.loads(path.read_text())
        mutations = [
            ("config", "batch_size", 2, "objective-only"),
            ("config", "initial_model_state_sha256", "f" * 64, "objective-only"),
            ("config", "resume_from_step", 0, "fresh runs"),
            (None, "sampled_batches_sha256", "f" * 64, "sampling differs"),
            (None, "sampling_state_sha256", "f" * 64, "sampling differs")]
        for container, key, value, error in mutations:
            with self.subTest(key=key):
                changed = json.loads(json.dumps(original))
                (changed[container] if container else changed)[key] = value
                path.write_text(json.dumps(changed))
                with self.assertRaisesRegex(ValueError, error):
                    compare_objectives(f.root / "l1", f.root / "mse", f.root / "rejected")
                self.assertFalse((f.root / "rejected").exists())
        path.write_text(json.dumps(original))

    def test_initial_fingerprint_tracks_weight_changes(self):
        model = torch.nn.Linear(2, 1)
        first = model_state_digest(model)
        self.assertEqual(first, model_state_digest(model))
        with torch.no_grad():
            model.weight.add_(1)
        self.assertNotEqual(first, model_state_digest(model))


if __name__ == "__main__":
    unittest.main()
