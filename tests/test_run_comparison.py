import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from virtual_staining.run_comparison import compare_duration


class DurationComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.base = self.root / "base"
        self.candidate = self.root / "candidate"
        self.config = {"steps": 2, "batch_size": 1, "patch_size": 8, "width": 2,
                       "seed": 17, "validate_every": 2, "device": "cpu",
                       "learning_rate": 0.001, "prepared_metadata_sha256": "data",
                       "pilot_plan_sha256": "plan", "depth_policy": "middle",
                       "git_revision": "same-source", "python": "3.12",
                       "packages": {"torch": "synthetic-fixture"}}
        self.create_run(self.base, self.config, 1.0)
        base = self.read(self.base)
        extended = self.config | {"steps": 4, "resume_from_step": 2,
                                  "resume_checkpoint_sha256": base["last_checkpoint_sha256"]}
        self.create_run(self.candidate, extended, 0.5)
        old_history = json.loads((self.base / "history.json").read_text())
        (self.candidate / "history.json").write_text(json.dumps(old_history + [{"step": 4}]))

    def create_run(self, directory, config, mae):
        directory.mkdir()
        # Synthetic opaque checkpoint bytes exercise artifact verification;
        # this report command intentionally does not load model tensors.
        hashes = {}
        for name in ("best", "last"):
            path = directory / f"{name}-model.pt"
            path.write_bytes((directory.name + name).encode())
            hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        fields = [{"source_id": "field-a", "group_id": "plate", "z": 3,
                   "mae": mae, "pearson": 0.2, "ssim": 0.1},
                  {"source_id": "field-b", "group_id": "plate", "z": 3,
                   "mae": mae, "pearson": None, "ssim": 0.1}]
        constant = [row | {"mae": 1.5, "pearson": None, "ssim": 0.05} for row in fields]
        summary = {"field_count": 2, "group_count": 1, "undefined_pearson_fields": 1,
                   "mae": mae, "pearson": 0.2, "ssim": 0.1}
        run = {"config": config, "test_pixels_loaded": False,
               "best_checkpoint_sha256": hashes["best"], "last_checkpoint_sha256": hashes["last"],
               "best_fields": fields, "best_model": summary, "constant_baseline_fields": constant,
               "constant_baseline": summary | {"mae": 1.5, "pearson": None, "ssim": 0.05,
                                                "undefined_pearson_fields": 2},
               "best_step": config["steps"], "train_fields": 8, "validation_fields": 2,
               "constant_value": 0.25, "constant_definition": "training median",
               "steps_this_run": config["steps"] - config.get("resume_from_step", 0)}
        (directory / "training.json").write_text(json.dumps(run))
        (directory / "history.json").write_text(json.dumps([{"step": 0}, {"step": config["steps"]}]))

    def read(self, directory):
        return json.loads((directory / "training.json").read_text())

    def write(self, directory, report):
        (directory / "training.json").write_text(json.dumps(report))

    def compare(self):
        return compare_duration(self.base, self.candidate, self.root / "comparison")

    def test_verified_continuation_reports_paired_deltas_and_undefined_correlation(self):
        result = self.compare()
        self.assertEqual(result["relative_mae_reduction"], 0.5)
        self.assertEqual(result["delta_candidate_minus_baseline"]["mae"], -0.5)
        self.assertEqual(result["field_deltas"][0]["delta_candidate_minus_baseline"]["pearson"], 0)
        self.assertIsNone(result["field_deltas"][1]["delta_candidate_minus_baseline"]["pearson"])
        self.assertFalse(result["test_pixels_loaded"])
        self.assertTrue((self.root / "comparison/comparison.json").exists())
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.compare()

    def test_zero_baseline_mae_has_no_relative_reduction(self):
        for directory in (self.base, self.candidate):
            run = self.read(directory)
            run["best_model"]["mae"] = 0
            for row in run["best_fields"]:
                row["mae"] = 0
            self.write(directory, run)
        self.assertIsNone(self.compare()["relative_mae_reduction"])

    def test_changed_settings_environment_source_or_extra_config_are_rejected(self):
        original = self.read(self.candidate)
        for key in ("device", "validate_every", "seed", "prepared_metadata_sha256",
                    "git_revision", "packages", "unexpected_setting"):
            with self.subTest(key=key):
                changed = json.loads(json.dumps(original))
                changed["config"][key] = "different"
                self.write(self.candidate, changed)
                with self.assertRaisesRegex(ValueError, "duration-only"):
                    self.compare()
                self.assertFalse((self.root / "comparison").exists())
        self.write(self.candidate, original)

    def test_modified_checkpoint_and_wrong_parent_are_rejected(self):
        run = self.read(self.candidate)
        run["config"]["resume_checkpoint_sha256"] = "wrong-parent"
        self.write(self.candidate, run)
        with self.assertRaisesRegex(ValueError, "directly continue"):
            self.compare()
        with (self.candidate / "last-model.pt").open("ab") as stream:
            stream.write(b"changed")
        with self.assertRaisesRegex(ValueError, "checksum"):
            self.compare()

    def test_changed_history_is_rejected(self):
        (self.candidate / "history.json").write_text(json.dumps([{"step": 0}, {"step": 4}]))
        with self.assertRaisesRegex(ValueError, "preserve.*history"):
            self.compare()

    def test_tampered_summary_duplicate_fields_and_nonfinite_metrics_are_rejected(self):
        original = self.read(self.candidate)
        changed = json.loads(json.dumps(original))
        changed["best_model"]["mae"] = 0.1
        self.write(self.candidate, changed)
        with self.assertRaisesRegex(ValueError, "summary disagrees"):
            self.compare()
        changed = json.loads(json.dumps(original))
        changed["best_fields"][1] = changed["best_fields"][0]
        self.write(self.candidate, changed)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.compare()
        changed = json.loads(json.dumps(original))
        changed["best_fields"][0]["mae"] = float("nan")
        self.write(self.candidate, changed)
        with self.assertRaisesRegex(ValueError, "non-finite"):
            self.compare()

    def test_changed_depth_and_test_access_are_rejected(self):
        run = self.read(self.candidate)
        for key in ("best_fields", "constant_baseline_fields"):
            run[key][0]["z"] = 4
        self.write(self.candidate, run)
        with self.assertRaisesRegex(ValueError, "depth changed"):
            self.compare()
        run["test_pixels_loaded"] = True
        self.write(self.candidate, run)
        with self.assertRaisesRegex(ValueError, "unopened test"):
            self.compare()

    def test_actual_trainer_resume_reports_are_accepted(self):
        try:
            import torch
            from test_trainer import TrainerTests
        except ImportError:
            self.skipTest("install the ml extra for end-to-end comparison")
        previous = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, previous)
        case = TrainerTests()
        fixture = case.fixture()
        self.addCleanup(fixture.temp.cleanup)
        case.run_training(fixture, "first", 2)
        case.run_training(fixture, "resumed", 4, fixture.root / "first/last-model.pt")
        result = compare_duration(fixture.root / "first", fixture.root / "resumed",
                                  fixture.root / "comparison")
        self.assertEqual(result["candidate_total_steps"], 4)


if __name__ == "__main__":
    unittest.main()
