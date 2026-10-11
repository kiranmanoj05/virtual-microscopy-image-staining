import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import tifffile

from virtual_staining.pilot import BASE_URL, select_pilot, write_pilot
from virtual_staining.pilot_audit import audit_pilot
from virtual_staining.paired_planes import load_pair


class PilotAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.raw = self.root / "raw"
        self.raw.mkdir()
        inventory = {"metadata_sha256": "a" * 64, "cell_line": "AICS-22", "fields": [
            {"source_id": f"allen-fov:{i}", "fov_id": str(i), "plate_id": str(i),
             "group_id": f"allen-plate:{i}", "well_id": str(i),
             "source_path": f"fovs/{i}.ome.tiff", "input_channel": 2, "target_channel": 1}
            for i in range(3)
        ]}
        self.plan = select_pilot(inventory, groups=3, fields_per_group=1, seed=17)
        for f in self.plan["fields"]:
            self.make_tiff(f)
            f["version_id"] = "v1"
            f["download_url"] = BASE_URL + f["source_path"] + "?versionId=v1"
            f["size_bytes"] = self.file_path(f).stat().st_size
        self.freeze()

    def file_path(self, field):
        return self.raw / f"allen_fov_{field['fov_id']}.ome.tiff"

    def make_tiff(self, field, target="Hoechst 33342"):
        pixels = np.arange(4 * 3 * 8 * 8, dtype=np.uint16).reshape(4, 3, 8, 8)
        tifffile.imwrite(self.file_path(field), pixels, photometric="minisblack",
                         metadata={"axes": "ZCYX", "Channel": {"Fluor": ["EGFP", target, "TL Brightfield"]}})

    def freeze(self):
        self.plan.pop("plan_sha256", None)
        self.plan["total_size_bytes"] = sum(f["size_bytes"] for f in self.plan["fields"])
        canonical = json.dumps(self.plan, sort_keys=True, separators=(",", ":"))
        self.plan["plan_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
        for f in self.plan["fields"]:
            receipt = {key: f[key] for key in ("source_id", "group_id", "split", "download_url", "size_bytes")}
            receipt["sha256"] = hashlib.sha256(self.file_path(f).read_bytes()).hexdigest()
            receipt["pilot_plan_sha256"] = self.plan["plan_sha256"]
            (self.raw / f"allen_fov_{f['fov_id']}.json").write_text(json.dumps(receipt))
        plan_dir = self.root / "plan"
        if not plan_dir.exists():
            write_pilot(self.plan, plan_dir)
        else:
            (plan_dir / "pilot.json").write_text(json.dumps(self.plan))
        self.plan_path = plan_dir / "pilot.json"

    def test_real_tiffs_decode_development_only_and_export_training_preview(self):
        with patch("virtual_staining.paired_planes.load_pair", wraps=load_pair) as loader:
            report = audit_pilot(self.plan_path, self.raw, self.root / "audit")
        self.assertTrue(report["valid"])
        self.assertFalse(report["test_pixels_loaded"])
        self.assertEqual(loader.call_count, 6)
        test_fov = next(f["fov_id"] for f in self.plan["fields"] if f["split"] == "test")
        self.assertTrue(all(f"allen_fov_{test_fov}.ome.tiff" not in str(call.args[0]) for call in loader.call_args_list))
        self.assertTrue((self.root / "audit/training-preview/preview.png").exists())

    def test_wrong_target_metadata_is_rejected_even_with_correct_receipt(self):
        f = self.plan["fields"][0]
        self.make_tiff(f, target="EGFP")
        f["size_bytes"] = self.file_path(f).stat().st_size
        self.freeze()
        result = audit_pilot(self.plan_path, self.raw, self.root / "audit")
        self.assertFalse(result["valid"])
        self.assertIn(f["source_id"], result["failed_sources"])

    def test_wrong_receipt_split_is_rejected(self):
        f = self.plan["fields"][0]
        path = self.raw / f"allen_fov_{f['fov_id']}.json"
        receipt = json.loads(path.read_text())
        receipt["split"] = "incorrect"
        path.write_text(json.dumps(receipt))
        self.assertFalse(audit_pilot(self.plan_path, self.raw, self.root / "audit")["valid"])

    def test_changed_bytes_fail_before_metadata_decode(self):
        f = self.plan["fields"][0]
        with self.file_path(f).open("ab") as stream:
            stream.write(b"changed")
        report = audit_pilot(self.plan_path, self.raw, self.root / "audit")
        self.assertFalse(report["valid"])
        full = json.loads((self.root / "audit/audit.json").read_text())
        self.assertIn("checksum", full["fields"][0]["errors"][0])
        self.assertNotIn("tiff", full["fields"][0])

    def test_existing_review_is_not_overwritten(self):
        output = self.root / "audit"
        output.mkdir()
        with self.assertRaisesRegex(ValueError, "already exists"):
            audit_pilot(self.plan_path, self.raw, output)


if __name__ == "__main__":
    unittest.main()
