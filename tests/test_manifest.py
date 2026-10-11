import csv
import tempfile
import unittest
from pathlib import Path

from virtual_staining.manifest import FIELDS, audit_manifest, file_hash


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.rows = []
        # Distinct byte fixtures check bookkeeping, not microscopy accuracy.
        for i, split in enumerate(("train", "validation", "test")):
            row = dict(pair_id=f"pair-{i}", source_id=f"field-{i}",
                       group_id=f"well-{i}", structure="example-target", split=split)
            for role in ("input", "target"):
                path = self.root / f"{i}-{role}.bin"
                path.write_bytes(f"fixture {i} {role}".encode())
                row[f"{role}_path"] = path.name
                row[f"{role}_sha256"] = file_hash(path)
            self.rows.append(row)

    def audit(self):
        manifest = self.root / "manifest.csv"
        with manifest.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(self.rows)
        return audit_manifest(manifest, self.root)

    def assert_error(self, message):
        result = self.audit()
        self.assertFalse(result["valid"])
        self.assertTrue(any(message in error for error in result["errors"]), result)

    def test_distinct_groups_and_files_pass(self):
        result = self.audit()
        self.assertTrue(result["valid"])
        self.assertEqual(result["counts"], {"train": 1, "validation": 1, "test": 1})

    def test_same_group_cannot_cross_splits(self):
        self.rows[1]["group_id"] = self.rows[0]["group_id"]
        self.assert_error("group_id well-0 crosses splits")

    def test_source_overlap_is_caught_even_with_different_groups(self):
        self.rows[1]["source_id"] = self.rows[0]["source_id"]
        self.assert_error("source_id field-0 crosses splits")

    def test_renamed_duplicate_content_cannot_cross_splits(self):
        first = self.root / self.rows[0]["input_path"]
        duplicate = self.root / self.rows[1]["input_path"]
        duplicate.write_bytes(first.read_bytes())
        self.rows[1]["input_sha256"] = file_hash(duplicate)
        self.assert_error("duplicate file content crosses splits")

    def test_duplicate_pair_ids_are_rejected(self):
        self.rows[1]["pair_id"] = self.rows[0]["pair_id"]
        self.assert_error("duplicate pair_id")

    def test_changed_bytes_are_rejected(self):
        (self.root / self.rows[0]["target_path"]).write_bytes(b"changed")
        self.assert_error("target checksum mismatch")

    def test_missing_file_is_rejected(self):
        (self.root / self.rows[0]["input_path"]).unlink()
        self.assert_error("missing input file")

    def test_path_cannot_escape_data_root(self):
        self.rows[0]["input_path"] = "../outside.bin"
        self.assert_error("path must stay inside data root")

    def test_unknown_split_is_rejected(self):
        self.rows[0]["split"] = "testing"
        self.assert_error("unknown split")

    def test_blank_group_is_rejected(self):
        self.rows[0]["group_id"] = ""
        self.assert_error("empty required field")

    def test_empty_manifest_is_not_a_successful_audit(self):
        self.rows = []
        self.assert_error("no observations")


if __name__ == "__main__":
    unittest.main()
