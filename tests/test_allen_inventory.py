import csv
import tempfile
import unittest
from pathlib import Path

from virtual_staining.allen_inventory import REQUIRED, OPTIONAL, read_allen_inventory, write_allen_inventory
from virtual_staining.splits import read_inventory


class AllenInventoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "metadata.csv"
        self.row = dict(zip(REQUIRED, ("1", "100", "20", "AICS-22", "fovs/a.ome.tiff", "6", "5")))

    def write(self, rows):
        with self.path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=REQUIRED + OPTIONAL)
            writer.writeheader()
            writer.writerows(rows)

    def read(self):
        return read_allen_inventory(self.path, cell_line="AICS-22")

    def test_cell_rows_collapse_into_fields_and_keep_plate_boundaries(self):
        second = self.row | {"FOVId": "2", "WellId": "21", "SourceReadPath": "fovs/b.ome.tiff"}
        other_line = self.row | {"CellLine": "AICS-12"}
        self.write([self.row, self.row, second, other_line])
        result = self.read()
        self.assertEqual(result["field_count"], 2)
        self.assertEqual(result["matched_cell_rows"], 3)
        self.assertEqual(result["plate_count"], 1)
        self.assertEqual(result["well_count"], 2)
        self.assertEqual({r["group_id"] for r in result["fields"]}, {"allen-plate:100"})

    def test_conflicting_channels_or_groups_within_a_field_fail(self):
        for key, value in [("ChannelNumber405", "4"), ("PlateId", "101"), ("WellId", "22")]:
            with self.subTest(key=key):
                self.write([self.row, self.row | {key: value}])
                with self.assertRaisesRegex(ValueError, "conflicting metadata"):
                    self.read()

    def test_shared_file_under_distinct_fov_ids_fails(self):
        self.write([self.row, self.row | {"FOVId": "2"}])
        with self.assertRaisesRegex(ValueError, "shared by distinct"):
            self.read()

    def test_missing_invalid_channels_and_unsafe_paths_fail(self):
        for change in [{"WellId": ""}, {"ChannelNumber405": "-1"},
                       {"ChannelNumber405": "6"}, {"ChannelNumber405": "5.0"},
                       {"SourceReadPath": "fovs/../a.tiff"}, {"SourceReadPath": "/fovs/a.tiff"}]:
            with self.subTest(change=change):
                self.write([self.row | change])
                with self.assertRaises(ValueError):
                    self.read()

    def test_no_matching_line_fails(self):
        self.write([self.row | {"CellLine": "AICS-12"}])
        with self.assertRaisesRegex(ValueError, "no fields found"):
            self.read()

    def test_export_can_feed_split_reader_and_refuses_overwrite(self):
        self.write([self.row])
        output = Path(self.temp.name) / "review"
        report = self.read()
        result = write_allen_inventory(report, output)
        self.assertEqual(read_inventory(output / "sources.csv"),
                         [{"source_id": "allen-fov:1", "group_id": "allen-plate:100"}])
        self.assertEqual(len(result["metadata_sha256"]), 64)
        with self.assertRaises(FileExistsError):
            write_allen_inventory(report, output)


if __name__ == "__main__":
    unittest.main()
