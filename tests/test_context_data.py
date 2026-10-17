import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    import numpy as np
    import torch
    import tifffile
    from virtual_staining.context_data import prepare_context_data, load_context_data
    from virtual_staining.paired_planes import load_context
    from virtual_staining.training_data import prepare_training_data, scale_plane
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "install the ml extra for context checks")
class ContextTests(unittest.TestCase):
    def fixture(self):
        from test_training import TrainingTests
        f = TrainingTests().fixture()
        self.addCleanup(f.temp.cleanup)
        prepare_training_data(f.root / 'audit/audit.json', f.root / 'baseline')
        records = [json.loads((f.raw / f"allen_fov_{field['fov_id']}.json").read_text())
                   for field in f.plan['fields']]
        f.receipts = f.root / 'receipts.json'
        f.receipts.write_text(json.dumps({'pilot_plan_sha256': f.plan['plan_sha256'], 'records': records}))
        return f

    def prepare(self, f):
        return prepare_context_data(f.plan_path, f.root / 'baseline/dataset.json', f.receipts,
                                    f.raw, f.root / 'context')

    def rehash(self, path, key):
        doc = json.loads(path.read_text())
        doc.pop(key, None)
        doc[key] = hashlib.sha256(json.dumps(doc, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        path.write_text(json.dumps(doc))

    def test_four_pages_only_and_czyx_axis_order(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'planes.ome.tiff'
            pixels = np.arange(3 * 4 * 8 * 12, dtype=np.uint16).reshape(3, 4, 8, 12)
            tifffile.imwrite(path, pixels, ome=True, photometric='minisblack', metadata={'axes': 'CZYX'})
            original = tifffile.TiffPage.asarray
            decoded = []
            def read(page, *args, **kwargs):
                decoded.append(page.index)
                return original(page, *args, **kwargs)
            with patch.object(tifffile.TiffPage, 'asarray', read):
                pair = load_context(path, z=2, input_channel=2, target_channel=1)
            np.testing.assert_array_equal(pair.input, pixels[2, 1:4])
            np.testing.assert_array_equal(pair.target, pixels[1, 2])
            self.assertEqual(decoded, [9, 10, 11, 6])
            self.assertEqual(pair.metadata['pages_decoded'], 4)

    def test_invalid_neighbourhood_rejects_before_any_decode(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'planes.ome.tiff'
            tifffile.imwrite(path, np.zeros((3, 2, 8, 8), dtype=np.uint16), ome=True,
                             photometric='minisblack', metadata={'axes': 'ZCYX'})
            with patch.object(tifffile.TiffPage, 'asarray', side_effect=AssertionError('decoded')):
                for z in (0, 2):
                    with self.assertRaisesRegex(ValueError, 'z index'):
                        load_context(path, z=z, input_channel=0, target_channel=1)

    def test_raw_controls_and_scaling_are_preserved_with_no_test_file_access(self):
        f = self.fixture()
        test_field = next(v for v in f.plan['fields'] if v['split'] == 'test')
        # No bytes or even a receipt are needed for the unopened test field.
        f.file_path(test_field).unlink()
        (f.raw / f"allen_fov_{test_field['fov_id']}.json").unlink()
        report = self.prepare(f)
        self.assertFalse(report['test_pixels_loaded'])
        self.assertEqual(report['field_count'], 2)
        meta, _, raw = load_context_data(f.root / 'context/context_dataset.json', scaled=False)
        _, _, scaled = load_context_data(f.root / 'context/context_dataset.json')
        baseline = json.loads((f.root / 'baseline/dataset.json').read_text())
        self.assertEqual(meta['scaling'], baseline['scaling'])
        for (field, context, target), (_, scaled_context, scaled_target) in zip(raw, scaled):
            old = next(b for b in baseline['fields'] if b['source_id'] == field['source_id'])
            np.testing.assert_array_equal(context[1], np.load(f.root / 'baseline' / old['input_file']))
            np.testing.assert_array_equal(target, np.load(f.root / 'baseline' / old['target_file']))
            np.testing.assert_array_equal(scaled_context, scale_plane(context, baseline['scaling']['input']))
            np.testing.assert_array_equal(scaled_target, scale_plane(target, baseline['scaling']['target']))
            self.assertEqual(field['target_sha256'], old['target_sha256'])

    def test_changed_raw_bytes_rejected_before_decode_and_output_creation(self):
        f = self.fixture()
        field = next(v for v in f.plan['fields'] if v['split'] != 'test')
        with f.file_path(field).open('ab') as stream:
            stream.write(b'changed')
        with patch('virtual_staining.context_data.load_context', side_effect=AssertionError('decoded')):
            with self.assertRaisesRegex(ValueError, 'checksum'):
                self.prepare(f)
        self.assertFalse((f.root / 'context').exists())

    def test_valid_but_wrong_frozen_target_is_rejected(self):
        f = self.fixture()
        path = f.root / 'baseline/dataset.json'
        baseline = json.loads(path.read_text())
        field = baseline['fields'][0]
        target = path.parent / field['target_file']
        np.save(target, np.load(target) + 1, allow_pickle=False)
        field['target_sha256'] = hashlib.sha256(target.read_bytes()).hexdigest()
        path.write_text(json.dumps(baseline))
        self.rehash(path, 'prepared_metadata_sha256')
        with self.assertRaisesRegex(ValueError, 'centre input or target'):
            self.prepare(f)

    def test_changed_receipt_version_and_split_controls_are_rejected(self):
        for key, value in [('download_url', 'wrong'), ('split', 'test'), ('sha256', '0' * 64)]:
            with self.subTest(key=key):
                f = self.fixture()
                receipts = json.loads(f.receipts.read_text())
                next(r for r in receipts['records'] if r['split'] == 'train')[key] = value
                f.receipts.write_text(json.dumps(receipts))
                with self.assertRaises(ValueError):
                    self.prepare(f)

    def test_rehashed_plan_with_changed_assignments_is_rejected(self):
        f = self.fixture()
        plan = json.loads(f.plan_path.read_text())
        next(v for v in plan['fields'] if v['split'] == 'train')['split'] = 'validation'
        f.plan_path.write_text(json.dumps(plan))
        self.rehash(f.plan_path, 'plan_sha256')
        with self.assertRaisesRegex(ValueError, 'plan or middle-depth'):
            self.prepare(f)

    def test_prepared_checksum_contract_and_centre_tampering_are_rejected(self):
        f = self.fixture()
        self.prepare(f)
        path = f.root / 'context/context_dataset.json'
        meta = json.loads(path.read_text())
        field = meta['fields'][0]
        pixels = path.parent / field['context_file']
        context = np.load(pixels)
        context[1] += 1
        np.save(pixels, context, allow_pickle=False)
        with self.assertRaisesRegex(ValueError, 'checksum'):
            load_context_data(path)
        field['context_sha256'] = hashlib.sha256(pixels.read_bytes()).hexdigest()
        path.write_text(json.dumps(meta))
        self.rehash(path, 'context_metadata_sha256')
        with self.assertRaisesRegex(ValueError, 'centre-input'):
            load_context_data(path)

    def test_output_is_not_overwritten(self):
        f = self.fixture()
        self.prepare(f)
        with self.assertRaisesRegex(ValueError, 'already exists'):
            self.prepare(f)


if __name__ == '__main__':
    unittest.main()
