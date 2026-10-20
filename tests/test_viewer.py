import base64
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from virtual_staining.viewer import ARMS, _scores, export_viewer, png_data, verify_saved_evaluation


class ViewerTests(unittest.TestCase):
    def fixture(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        protocol = {'test_pixels_loaded': False, 'input_offsets': [-1, 0, 1], 'target_offset': 0,
                    'test_fields': [{'source_id': 'f1', 'group_id': 'g1'}, {'source_id': 'f2', 'group_id': 'g1'}],
                    'validation_selected_context': 'neighbouring', 'constant_value': .25,
                    'scaling': {'input': {'low': 0., 'high': 100., 'clip': True},
                                'target': {'low': 0., 'high': 50., 'clip': False}, 'fitted_on': 'train only'},
                    'models': []}
        identity = hashlib.sha256(json.dumps(protocol, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        report = {'schema': 'locked-heldout-evaluation-v1', 'test_pixels_loaded': True, 'protocol': protocol,
                  'protocol_sha256': identity, 'validation_selected_context': 'neighbouring',
                  'artifacts': [], 'fields': {a: [] for a in ARMS}, 'summaries': {}, 'limitations': ['One test plate.']}
        for i in range(2):
            raw = np.arange(8*12, dtype=np.uint16).reshape(8, 12) + i
            target = raw.astype(np.float32) / 50
            arrays = {'raw-target': raw, 'raw-context': np.stack([raw]*3), 'input': np.stack([raw]*3).astype(np.float32)/100,
                      'target': target, 'repeated-centre': target*.3, 'neighbouring': target*.5,
                      'constant': np.full_like(target, .25)}
            artifact = {'source_id': f'f{i+1}', 'group_id': 'g1', 'split': 'test', 'z': 2, 'arrays': {}}
            for name, array in arrays.items():
                path = root / f'{i}-{name}.npy'
                np.save(path, array, allow_pickle=False)
                artifact['arrays'][name] = {'file': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
            report['artifacts'].append(artifact)
            for arm in ARMS:
                report['fields'][arm].append({k: artifact[k] for k in ('source_id', 'group_id', 'split', 'z')} | {
                    **_scores(arrays[arm], target), 'bright_pixel_count': int((target >= 1).sum())})
        for arm in ARMS:
            rows = report['fields'][arm]
            report['summaries'][arm] = {'field_count': 2, 'group_count': 1,
                                       'undefined_pearson_fields': sum(r['pearson'] is None for r in rows),
                                       'undefined_bright_fields': 0}
            for key in ('mae', 'pearson', 'ssim', 'bright_pixel_mae'):
                values = [r[key] for r in rows if r[key] is not None]
                report['summaries'][arm][key] = float(np.mean(values)) if values else None
        path = root/'evaluation.json'
        path.write_text(json.dumps(report))
        return path, identity, report

    def test_export_includes_all_fields_arms_and_offline_images_and_preserves_source(self):
        path, sha, _ = self.fixture()
        before = path.read_bytes()
        out = path.parent/'viewer.html'
        result = export_viewer(path, out, expected_protocol_sha256=sha)
        self.assertEqual(result['field_count'], 2)
        self.assertFalse(result['inference_performed'])
        self.assertEqual(path.read_bytes(), before)
        html = out.read_text()
        self.assertIn('Content-Security-Policy', html)
        self.assertIn('data:image/png;base64,', html)
        self.assertIn('validation-selected', html)
        self.assertIn('Useful staining is not established', html)
        self.assertNotIn('https://', html)
        self.assertNotIn('__EVIDENCE_JSON__', html)
        with self.assertRaisesRegex(ValueError, 'already exists'):
            export_viewer(path, out, expected_protocol_sha256=sha)

    def test_changed_protocol_and_missing_duplicate_test_fields_fail(self):
        for change in ('sha', 'missing', 'duplicate'):
            path, sha, report = self.fixture()
            if change == 'sha': sha = '0'*64
            elif change == 'missing': report['artifacts'].pop()
            else: report['artifacts'][1] = report['artifacts'][0]
            path.write_text(json.dumps(report))
            with self.assertRaises(ValueError): verify_saved_evaluation(path, sha)

    def test_array_corruption_and_metric_tampering_fail(self):
        for change in ('pixels', 'metric', 'summary', 'depth'):
            path, sha, report = self.fixture()
            if change == 'pixels': (path.parent/report['artifacts'][0]['arrays']['target']['file']).write_bytes(b'bad')
            elif change == 'metric': report['fields']['neighbouring'][0]['ssim'] += .01
            elif change == 'summary': report['summaries']['constant']['undefined_pearson_fields'] = 0
            else: report['fields']['constant'][0]['z'] += 1
            path.write_text(json.dumps(report))
            with self.assertRaises(ValueError): verify_saved_evaluation(path, sha)

    def test_unsafe_and_symlink_array_paths_rejected(self):
        for name in ('../target.npy', '/tmp/target.npy'):
            path, sha, report = self.fixture()
            report['artifacts'][0]['arrays']['target']['file'] = name
            path.write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, 'path/checksum'): verify_saved_evaluation(path, sha)
        path, sha, report = self.fixture()
        item = report['artifacts'][0]['arrays']['target']
        target = path.parent/item['file']
        outside = path.parent.parent/(path.parent.name+'-outside.npy')
        outside.write_bytes(target.read_bytes()); self.addCleanup(outside.unlink)
        target.unlink(); target.symlink_to(outside)
        with self.assertRaisesRegex(ValueError, 'path/checksum'): verify_saved_evaluation(path, sha)

    def test_rehashed_wrong_scaling_and_nonfinite_arrays_rejected(self):
        for mutation in ('scaling', 'nan', 'geometry', 'constant'):
            path, sha, report = self.fixture()
            key = 'input' if mutation == 'scaling' else 'constant' if mutation == 'constant' else 'neighbouring'
            item = report['artifacts'][0]['arrays'][key]; source = path.parent/item['file']
            data = np.load(source)
            if mutation == 'nan': data[0,0] = np.nan
            elif mutation == 'geometry': data = data[:4]
            else: data += .1
            np.save(source, data); item['sha256'] = hashlib.sha256(source.read_bytes()).hexdigest()
            path.write_text(json.dumps(report))
            with self.assertRaises(ValueError): verify_saved_evaluation(path, sha)

    def test_png_scaling_is_fixed_and_clips_display_only(self):
        a = np.array([[-1., 0., .5, 1., 2.]])
        before = a.copy()
        png = png_data(a, 'gray')
        im = np.array(Image.open(io.BytesIO(base64.b64decode(png.split(',')[1]))))
        np.testing.assert_array_equal(a, before)
        np.testing.assert_array_equal(im[0,0], im[0,1])
        np.testing.assert_array_equal(im[0,3], im[0,4])
        self.assertTrue(120 <= im[0,2,0] <= 130)

    def test_report_identifiers_cannot_close_script_tag(self):
        path, sha, report = self.fixture()
        report['limitations'] = ['</script><img src=x onerror=alert(1)>']
        path.write_text(json.dumps(report)); out = path.parent/'viewer.html'
        export_viewer(path, out, expected_protocol_sha256=sha)
        html = out.read_text()
        self.assertNotIn('</script><img', html)
        self.assertIn('\\u003c/script', html)

    def test_cli_error_does_not_create_output(self):
        from virtual_staining.cli import main
        path, sha, report = self.fixture(); report['artifacts'].pop(); path.write_text(json.dumps(report))
        out = path.parent/'no.html'
        self.assertEqual(main(['export-viewer', str(path), '--protocol-sha256', sha, '--output', str(out)]), 2)
        self.assertFalse(out.exists())
