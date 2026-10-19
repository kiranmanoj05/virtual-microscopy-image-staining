import hashlib
import json
import unittest
from unittest.mock import patch

try:
    import numpy as np
    import torch
    from virtual_staining.evaluation import evaluate_heldout, verify_evaluation, METRICS
    from virtual_staining.model import UNet
    from virtual_staining.paired_planes import load_context
    from virtual_staining.training_data import normalisation_fingerprint
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'install ml dependencies')
class EvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def fixture(self):
        from test_context_data import ContextTests
        helper = ContextTests()
        f = helper.fixture()
        self.addCleanup(helper.doCleanups)
        scaling = json.loads((f.root / 'baseline/dataset.json').read_text())['scaling']
        f.lock = {'protocol_version': 1, 'status': 'locked before test pixels opened',
                  'pilot_plan_sha256': f.plan['plan_sha256'], 'development_context_sha256': 'a' * 64,
                  'experiment_source': 'test-source', 'models': [], 'validation_selected_context': 'neighbouring',
                  'test_fields': [{k: field[k] for k in ('source_id','group_id','fov_id','version_id')}
                                  for field in f.plan['fields'] if field['split'] == 'test'],
                  'depth_policy': 'floor(SizeZ/2), fixed for every field', 'input_offsets': [-1,0,1],
                  'target_offset': 0, 'input_channel': 2, 'target_channel': 1, 'scaling': scaling,
                  'constant_value': 0.2, 'metrics': METRICS, 'test_pixels_loaded': False}
        torch.manual_seed(17)
        model = UNet(width=2, input_channels=3)
        for policy, mae in [('repeated-centre', .4), ('neighbouring', .3)]:
            config = {'width': 2, 'input_context': policy, 'input_channels': 3,
                      'input_normalisation': 'global', 'pilot_plan_sha256': f.plan['plan_sha256'],
                      'prepared_metadata_sha256': 'a' * 64, 'git_revision': 'test-source',
                      'depth_policy': f.lock['depth_policy'],
                      'preprocessing_sha256': normalisation_fingerprint(f.lock, 'global'),
                      'ssim_data_range_scaled': 1.0, 'ssim_window': 7,
                      'parameter_count': sum(p.numel() for p in model.parameters())}
            path = f.root / f'{policy}.pt'
            torch.save({'model_state': model.state_dict(), 'step': 2, 'config': config, 'scaling': scaling}, path)
            f.lock['models'].append({'input_context': policy, 'checkpoint_path': path.name,
                                     'checkpoint_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                                     'step': 2, 'parameter_count': config['parameter_count'], 'validation_mae': mae})
        f.protocol = f.root / 'protocol.json'
        self.freeze(f)
        return f

    def freeze(self, f):
        f.lock.pop('protocol_sha256', None)
        f.identity = hashlib.sha256(json.dumps(f.lock, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        f.lock['protocol_sha256'] = f.identity
        f.protocol.write_text(json.dumps(f.lock))

    def run_evaluation(self, f):
        return evaluate_heldout(f.protocol, f.plan_path, f.receipts, f.root, f.raw,
                               f.root / 'evaluation', expected_protocol_sha256=f.identity)

    def test_complete_locked_evaluation_decodes_only_test_four_pages_and_saves_exact_metrics(self):
        f = self.fixture()
        with patch('virtual_staining.evaluation.load_context', wraps=load_context) as reader:
            result = self.run_evaluation(f)
        self.assertTrue(result['test_pixels_loaded'])
        self.assertEqual(reader.call_count, 1)
        self.assertEqual(reader.call_args.args[0], f.file_path(next(v for v in f.plan['fields'] if v['split'] == 'test')))
        report = json.loads((f.root / 'evaluation/evaluation.json').read_text())
        self.assertEqual(report['provenance'][0]['planes']['pages_decoded'], 4)
        files = report['artifacts'][0]['arrays']
        target = np.load(f.root / 'evaluation' / files['target']['file'])
        for policy in ('repeated-centre', 'neighbouring', 'constant'):
            prediction = np.load(f.root / 'evaluation' / files[policy]['file'])
            self.assertEqual(float(np.abs(prediction.astype(np.float64) - target).mean()), report['fields'][policy][0]['mae'])
        self.assertIsNone(report['summaries']['constant']['pearson'])
        self.assertEqual(report['summaries']['constant']['undefined_pearson_fields'], 1)
        self.assertEqual(len(list((f.root / 'evaluation').rglob('test-preview.png'))), 3)
        with self.assertRaisesRegex(ValueError, 'already exists'):
            self.run_evaluation(f)

    def test_protocol_changes_and_rehashed_choices_need_original_external_fingerprint(self):
        f = self.fixture()
        original = f.identity
        f.lock['constant_value'] += 1
        self.freeze(f)
        f.identity = original
        with patch('virtual_staining.evaluation.load_context', side_effect=AssertionError('decoded')):
            with self.assertRaisesRegex(ValueError, 'precommitted'):
                self.run_evaluation(f)
        self.assertFalse((f.root / 'evaluation').exists())

    def test_changed_checkpoint_rejected_before_pixels(self):
        f = self.fixture()
        (f.root / f.lock['models'][0]['checkpoint_path']).write_bytes(b'changed')
        with patch('virtual_staining.evaluation.load_context', side_effect=AssertionError('decoded')):
            with self.assertRaisesRegex(ValueError, 'checksum'):
                self.run_evaluation(f)

    def test_rehashed_wrong_checkpoint_contract_rejected(self):
        for key, value in [('input_context', 'single'), ('input_normalisation', 'per-plane'), ('ssim_window', 9)]:
            with self.subTest(key=key):
                f = self.fixture()
                path = f.root / f.lock['models'][0]['checkpoint_path']
                saved = torch.load(path, weights_only=True)
                saved['config'][key] = value
                torch.save(saved, path)
                f.lock['models'][0]['checkpoint_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
                self.freeze(f)
                with self.assertRaisesRegex(ValueError, 'configuration'):
                    self.run_evaluation(f)

    def test_raw_corruption_and_receipt_split_rejected_before_decode(self):
        for mutate in ('raw', 'receipt'):
            f = self.fixture()
            field = next(v for v in f.plan['fields'] if v['split'] == 'test')
            if mutate == 'raw':
                with f.file_path(field).open('ab') as stream:
                    stream.write(b'changed')
            else:
                receipt = json.loads(f.receipts.read_text())
                next(r for r in receipt['records'] if r['split'] == 'test')['split'] = 'train'
                f.receipts.write_text(json.dumps(receipt))
            with patch('virtual_staining.evaluation.load_context', side_effect=AssertionError('decoded')):
                with self.assertRaises(ValueError):
                    self.run_evaluation(f)
            self.assertFalse((f.root / 'evaluation').exists())

    def test_wrong_channel_metadata_rejected_before_decode(self):
        f = self.fixture()
        from virtual_staining.evaluation import read_tiff_info
        def wrong(path):
            info = read_tiff_info(path)
            info['ome_images'][0]['channels'][1]['Fluor'] = 'EGFP'
            return info
        with patch('virtual_staining.evaluation.read_tiff_info', side_effect=wrong), patch(
                'virtual_staining.evaluation.load_context', side_effect=AssertionError('decoded')):
            with self.assertRaisesRegex(ValueError, 'channels'):
                self.run_evaluation(f)

    def test_missing_or_duplicate_test_identity_rejected(self):
        for mutation in ('missing', 'duplicate'):
            f = self.fixture()
            if mutation == 'missing':
                f.lock['test_fields'] = []
            else:
                f.lock['test_fields'] *= 2
            self.freeze(f)
            with self.assertRaisesRegex(ValueError, 'identities'):
                self.run_evaluation(f)

    def test_inference_failure_does_not_commit_partial_output(self):
        f = self.fixture()
        with patch('virtual_staining.evaluation.image_scores', side_effect=ValueError('failed inference')):
            with self.assertRaisesRegex(ValueError, 'failed inference'):
                self.run_evaluation(f)
        self.assertFalse((f.root / 'evaluation').exists())
        self.assertEqual(list(f.root.glob('.heldout-*')), [])
