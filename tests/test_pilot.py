import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from virtual_staining.pilot import select_pilot, resolve_downloads, write_pilot, download_pilot


class Response(io.BytesIO):
    def __init__(self, payload=b"pixels", version="version-one"):
        super().__init__(payload)
        self.headers = {"x-amz-version-id": version, "Content-Length": str(len(payload))}


class PilotTests(unittest.TestCase):
    def setUp(self):
        self.inventory = {"metadata_sha256": "a" * 64, "cell_line": "AICS-22", "fields": [
            {"source_id": f"allen-fov:{i}", "fov_id": str(i),
             "plate_id": str(i // 3), "group_id": f"allen-plate:{i // 3}",
             "well_id": str(i), "source_path": f"fovs/{i}.ome.tiff",
             "input_channel": 6, "target_channel": 5}
            for i in range(18)
        ]}

    def plan(self):
        return select_pilot(self.inventory, groups=6, fields_per_group=2, seed=17)

    def pinned(self):
        with patch("virtual_staining.pilot.urllib.request.urlopen", side_effect=lambda *a, **k: Response()):
            return resolve_downloads(self.plan())

    def test_selection_is_order_independent_and_keeps_entire_plates(self):
        first = self.plan()
        self.inventory["fields"].reverse()
        self.assertEqual(first, self.plan())
        self.assertEqual(first["split_plan"]["source_counts"], {"test": 2, "train": 8, "validation": 2})
        for group in {f["group_id"] for f in first["fields"]}:
            fields = [f for f in first["fields"] if f["group_id"] == group]
            self.assertEqual(len({f["split"] for f in fields}), 1)
            self.assertEqual(len({f["well_id"] for f in fields}), 2)

    def test_insufficient_groups_or_distinct_wells_fail(self):
        with self.assertRaises(ValueError):
            select_pilot(self.inventory, groups=7, fields_per_group=2, seed=17)
        for f in self.inventory["fields"]:
            f["well_id"] = "one"
        with self.assertRaises(ValueError):
            self.plan()

    def test_missing_version_fails_before_plan_is_written(self):
        with patch("virtual_staining.pilot.urllib.request.urlopen", return_value=Response(version="null")):
            with self.assertRaisesRegex(ValueError, "S3 version"):
                resolve_downloads(self.plan())

    def test_download_receipts_support_verified_reruns_and_detect_corruption(self):
        plan = self.pinned()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_pilot(plan, root / "plan")
            path = root / "plan/pilot.json"
            with patch("virtual_staining.pilot.urllib.request.urlopen", side_effect=lambda *a, **k: Response()) as network:
                result = download_pilot(path, root / "raw")
                self.assertEqual(result["downloaded_or_verified_fields"], 12)
                self.assertEqual(network.call_count, 12)
                download_pilot(path, root / "raw")
                self.assertEqual(network.call_count, 12)
                Path(result["records"][0]["path"]).write_bytes(b"damage")
                with self.assertRaisesRegex(ValueError, "integrity check"):
                    download_pilot(path, root / "raw")

    def test_modified_plan_fails_without_network_access(self):
        plan = self.pinned()
        plan["fields"][0]["split"] = "validation" if plan["fields"][0]["split"] != "validation" else "train"
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "pilot.json"
            path.write_text(json.dumps(plan))
            with patch("virtual_staining.pilot.urllib.request.urlopen") as network:
                with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
                    download_pilot(path, Path(temp) / "raw")
                network.assert_not_called()

    def test_unexpected_version_or_short_download_is_not_promoted(self):
        plan = self.pinned()
        for response in [Response(version="wrong"), Response(payload=b"short")]:
            with self.subTest(response=response), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                write_pilot(plan, root / "plan")
                with patch("virtual_staining.pilot.urllib.request.urlopen", return_value=response):
                    with self.assertRaises(ValueError):
                        download_pilot(root / "plan/pilot.json", root / "raw")
                self.assertEqual(list((root / "raw").glob("*.ome.tiff")), [])

    def test_existing_unreceipted_image_is_not_overwritten(self):
        plan = self.pinned()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_pilot(plan, root / "plan")
            (root / "raw").mkdir()
            field = plan["fields"][0]
            (root / "raw" / f"allen_fov_{field['fov_id']}.ome.tiff").write_bytes(b"existing")
            with self.assertRaisesRegex(ValueError, "no receipt"):
                download_pilot(root / "plan/pilot.json", root / "raw")

    def test_development_only_preserves_plan_and_never_requests_test_objects(self):
        plan = self.pinned()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_pilot(plan, root / "plan")
            path = root / "plan/pilot.json"
            before = path.read_bytes()
            with patch("virtual_staining.pilot.urllib.request.urlopen", side_effect=lambda *a, **k: Response()) as network:
                result = download_pilot(path, root / "raw", development_only=True)
            test_urls = {f['download_url'] for f in plan['fields'] if f['split'] == 'test'}
            self.assertEqual(network.call_count, 10)
            self.assertTrue(all(call.args[0] not in test_urls for call in network.call_args_list))
            self.assertEqual(result['pilot_plan_sha256'], plan['plan_sha256'])
            self.assertTrue(result['development_only'])
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(len(list((root / 'raw').glob('*.ome.tiff'))), 10)
            self.assertTrue(all(r['split'] != 'test' for r in result['records']))


if __name__ == "__main__":
    unittest.main()
