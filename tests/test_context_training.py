import hashlib
import json
import unittest

try:
    import numpy as np
    import torch
    from virtual_staining.model import UNet
    from virtual_staining.trainer import load_training_inputs, train_pilot, sample_batch, model_state_digest
    from virtual_staining.diagnostics import diagnose_training
    from virtual_staining.context_comparison import compare_context
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'install ml extra for context training checks')
class ContextTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def fixture(self):
        from test_context_data import ContextTests
        t = ContextTests()
        f = t.fixture()
        self.addCleanup(f.temp.cleanup)
        t.prepare(f)
        f.dataset = f.root / 'context/context_dataset.json'
        return f

    def train(self, f, name, policy, steps=2, resume=None):
        return train_pilot(f.dataset, f.root/name, steps=steps, batch_size=1, patch_size=8,
                           width=2, seed=17, validate_every=2, objective='mse',
                           device_name='cpu', input_context=policy, resume_path=resume)

    def test_three_input_model_shape_gradients_and_seeded_weights_match(self):
        torch.manual_seed(17)
        a = UNet(2, input_channels=3)
        torch.manual_seed(17)
        b = UNet(2, input_channels=3)
        self.assertEqual(model_state_digest(a), model_state_digest(b))
        x = torch.rand(2,3,17,19)
        y = a(x)
        self.assertEqual(y.shape, (2,1,17,19))
        y.square().mean().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in a.parameters()))
        with self.assertRaises(ValueError):
            a(torch.zeros(1,1,8,8))

    def test_spatial_sampling_keeps_all_channels_and_target_aligned(self):
        target = np.arange(17*19,dtype=np.float32).reshape(17,19)
        inputs = np.stack([target+i*1000 for i in range(3)])
        trace = hashlib.sha256()
        x,y = sample_batch([({'source_id':'field'},inputs,target)], np.random.default_rng(17),
                            8,12,trace_digest=trace)
        self.assertEqual(x.shape,(12,3,8,8))
        for i in range(3):
            torch.testing.assert_close(x[:,i:i+1],y+i*1000,rtol=0,atol=0)

    def test_loader_preserves_targets_and_middle_only_information(self):
        f = self.fixture()
        _,a,centre = load_training_inputs(f.dataset,input_context='repeated-centre')
        _,b,neighbour = load_training_inputs(f.dataset,input_context='neighbouring')
        self.assertEqual(a,b)
        for left,right in zip(centre,neighbour):
            np.testing.assert_array_equal(left[2],right[2])
            for channel in range(3):
                np.testing.assert_array_equal(left[1][channel],right[1][1])
        with self.assertRaisesRegex(ValueError,'frozen global'):
            load_training_inputs(f.dataset,input_context='neighbouring',input_normalisation='per-plane')

    def test_matched_runs_and_diagnostics_restore_context_policy(self):
        f = self.fixture()
        a = self.train(f,'centre','repeated-centre')
        b = self.train(f,'neighbour','neighbouring')
        for key in ('sampler_trace_sha256','target_batches_sha256','sampling_state_sha256'):
            self.assertEqual(a[key],b[key])
        self.assertEqual(a['config']['initial_model_state_sha256'],b['config']['initial_model_state_sha256'])
        self.assertEqual(a['config']['parameter_count'],b['config']['parameter_count'])
        self.assertNotEqual(a['sampled_batches_sha256'],b['sampled_batches_sha256'])
        comparison = compare_context(f.root/'centre',f.root/'neighbour',f.root/'comparison')
        self.assertFalse(comparison['test_pixels_loaded'])
        diagnose_training(f.dataset,f.root/'neighbour',f.root/'diagnostics')
        report=json.loads((f.root/'diagnostics/diagnostics.json').read_text())
        self.assertEqual(report['input_context'],'neighbouring')
        self.assertEqual(report['checkpoints']['best']['split_scores']['validation']['model'],b['best_model'])
        path=f.root/'neighbour/training.json'
        run=json.loads(path.read_text());run['config']['input_context']='repeated-centre'
        path.write_text(json.dumps(run))
        with self.assertRaisesRegex(ValueError,'checkpoint input context'):
            diagnose_training(f.dataset,f.root/'neighbour',f.root/'rejected')

    def test_context_locked_resume_reproduces_uninterrupted_weights(self):
        f=self.fixture()
        self.train(f,'first','neighbouring')
        with self.assertRaisesRegex(ValueError,'input_context'):
            self.train(f,'wrong','repeated-centre',4,f.root/'first/last-model.pt')
        self.assertFalse((f.root/'wrong').exists())
        self.train(f,'resumed','neighbouring',4,f.root/'first/last-model.pt')
        self.train(f,'full','neighbouring',4)
        a=torch.load(f.root/'resumed/last-model.pt',weights_only=True)
        b=torch.load(f.root/'full/last-model.pt',weights_only=True)
        for key in a['model_state']:
            self.assertTrue(torch.equal(a['model_state'][key],b['model_state'][key]))

    def test_comparison_rejects_extra_changes_targets_sampling_and_misreported_checkpoints(self):
        f=self.fixture()
        self.train(f,'centre','repeated-centre')
        self.train(f,'neighbour','neighbouring')
        path=f.root/'neighbour/training.json'
        original=json.loads(path.read_text())
        for section,key,value in [('config','objective','l1'),('config','parameter_count',1),
                                  ('config','input_channels',1),('config','initial_model_state_sha256','0'*64),
                                  ('root','target_batches_sha256','0'*64),('root','sampler_trace_sha256','0'*64),
                                  ('root','steps_this_run',1)]:
            with self.subTest(key=key):
                changed=json.loads(json.dumps(original))
                (changed['config'] if section=='config' else changed)[key]=value
                path.write_text(json.dumps(changed))
                with self.assertRaises(ValueError):
                    compare_context(f.root/'centre',f.root/'neighbour',f.root/'rejected')
        self.assertFalse((f.root/'rejected').exists())


if __name__=='__main__':
    unittest.main()
