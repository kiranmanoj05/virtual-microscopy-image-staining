import csv
import json
import tempfile
import unittest
from pathlib import Path

from virtual_staining.splits import create_split_plan, read_inventory, write_split_plan


class SplitTests(unittest.TestCase):
    def setUp(self):
        self.rows = [
            {"source_id": f"field-{index}", "group_id": f"well-{index // 2}"}
            for index in range(10)
        ]

    def test_whole_groups_are_assigned_and_counts_are_explicit(self):
        plan = create_split_plan(self.rows, validation_groups=1, test_groups=1, seed=17)
        observed = {}
        for row in plan["assignments"]:
            observed.setdefault(row["group_id"], set()).add(row["split"])
        self.assertTrue(all(len(splits) == 1 for splits in observed.values()))
        self.assertEqual(plan["group_counts"], {"test": 1, "train": 3, "validation": 1})
        self.assertEqual(plan["source_counts"], {"test": 2, "train": 6, "validation": 2})

    def test_input_order_does_not_change_assignments_or_fingerprint(self):
        first = create_split_plan(self.rows, validation_groups=1, test_groups=1, seed=17)
        second = create_split_plan(list(reversed(self.rows)), validation_groups=1,
                                   test_groups=1, seed=17)
        self.assertEqual(first, second)

    def test_seed_change_changes_a_frozen_identity(self):
        first = create_split_plan(self.rows, validation_groups=1, test_groups=1, seed=17)
        second = create_split_plan(self.rows, validation_groups=1, test_groups=1, seed=18)
        self.assertNotEqual(first["fingerprint_sha256"], second["fingerprint_sha256"])

    def test_at_least_one_training_group_is_required(self):
        rows = self.rows[:4]
        with self.assertRaisesRegex(ValueError, "training group"):
            create_split_plan(rows, validation_groups=1, test_groups=1, seed=17)

    def test_inventory_rejects_duplicate_and_conflicting_sources(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "inventory.csv"
            path.write_text("source_id,group_id\nfov-1,well-a\nfov-1,well-b\n")
            with self.assertRaisesRegex(ValueError, "multiple groups"):
                read_inventory(path)
            path.write_text("source_id,group_id\nfov-1,well-a\nfov-1,well-a\n")
            with self.assertRaisesRegex(ValueError, "duplicate source_id"):
                read_inventory(path)

    def test_written_plan_is_fingerprinted_and_cannot_be_overwritten(self):
        plan = create_split_plan(self.rows, validation_groups=1, test_groups=1, seed=17)
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "frozen"
            result = write_split_plan(plan, output)
            with (output / "assignments.csv").open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            metadata = json.loads((output / "split.json").read_text())
            self.assertEqual(len(rows), 10)
            self.assertEqual(metadata["fingerprint_sha256"], plan["fingerprint_sha256"])
            self.assertEqual(result["assignments_file"], "assignments.csv")
            with self.assertRaises(FileExistsError):
                write_split_plan(plan, output)


if __name__ == "__main__":
    unittest.main()
