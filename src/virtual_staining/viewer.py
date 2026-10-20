"""Verify saved evaluation evidence and export an offline, inference-free viewer."""

import base64
import hashlib
import io
import json
from importlib.resources import files
from pathlib import Path

import numpy as np
from matplotlib import colormaps
from PIL import Image
from skimage.metrics import structural_similarity


ARMS = ('repeated-centre', 'neighbouring', 'constant')


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _scores(prediction, target):
    p, t = prediction.astype(np.float64).ravel(), target.astype(np.float64).ravel()
    p, t = p - p.mean(), t - t.mean()
    denominator = float(np.linalg.norm(p) * np.linalg.norm(t))
    bright = target >= 1.0
    return {'mae': float(np.abs(prediction.astype(np.float64) - target).mean()),
            'pearson': float(np.dot(p, t) / denominator) if denominator > 0 else None,
            'ssim': float(structural_similarity(target, prediction, data_range=1.0, win_size=7)),
            'bright_pixel_mae': float(np.abs(prediction[bright] - target[bright]).mean()) if bright.any() else None}


def _equal(left, right):
    if left is None or right is None:
        return left is right
    return bool(np.isfinite(left) and np.isfinite(right) and np.isclose(left, right, rtol=1e-7, atol=1e-9))


def verify_saved_evaluation(report_path: Path, expected_protocol_sha256: str):
    """No checkpoints, raw TIFFs, training code, Torch or network access required.

    Recompute the frozen metrics from hash-verified saved arrays. Never refit
    scaling or select an arm. Verification is evidence inspection, not inference.
    """
    report = json.loads(report_path.read_text())
    protocol = dict(report['protocol'])
    identity = protocol.pop('protocol_sha256', report['protocol_sha256'])
    canonical = json.dumps(protocol, sort_keys=True, separators=(',', ':')).encode()
    if (hashlib.sha256(canonical).hexdigest() != identity or identity != expected_protocol_sha256 or
            report['protocol_sha256'] != identity or report['schema'] != 'locked-heldout-evaluation-v1' or
            report['test_pixels_loaded'] is not True or protocol['test_pixels_loaded'] is not False):
        raise ValueError('saved protocol/report does not match the retained evaluation lock')
    if (protocol['input_offsets'] != [-1, 0, 1] or protocol['target_offset'] != 0 or
            protocol['scaling']['fitted_on'] != 'train only' or
            protocol['validation_selected_context'] not in ARMS[:2] or
            report['validation_selected_context'] != protocol['validation_selected_context']):
        raise ValueError('unsupported saved evaluation contract')
    expected = [(f['source_id'], f['group_id']) for f in protocol['test_fields']]
    actual = [(f['source_id'], f['group_id']) for f in report['artifacts']]
    if not expected or len(set(s for s, _ in expected)) != len(expected) or actual != expected:
        raise ValueError('viewer must include every locked test source exactly once')
    arrays_by_source = []
    for artifact in report['artifacts']:
        if artifact['split'] != 'test':
            raise ValueError('viewer accepts held-out artifacts only')
        arrays = {}
        for name in ('raw-context', 'raw-target', 'input', 'target', *ARMS):
            item = artifact['arrays'][name]
            path = report_path.parent / item['file']
            if (Path(item['file']).name != item['file'] or
                    not path.resolve().is_relative_to(report_path.parent.resolve()) or
                    _sha(path) != item['sha256']):
                raise ValueError('saved array basename/path/checksum mismatch')
            array = np.load(path, allow_pickle=False)
            if array.dtype.kind not in 'uif' or not np.isfinite(array).all():
                raise ValueError('saved arrays must be numeric and finite')
            arrays[name] = array
        target = arrays['target']
        if target.ndim != 2 or min(target.shape) < 8:
            raise ValueError('saved target must be YX with dimensions >=8')
        for name, array in arrays.items():
            expected_shape = (3, *target.shape) if name in ('input', 'raw-context') else target.shape
            if array.shape != expected_shape:
                raise ValueError('saved array geometry differs from target')
        for raw, scaled, key in [('raw-context', 'input', 'input'), ('raw-target', 'target', 'target')]:
            params = protocol['scaling'][key]
            if (not all(np.isfinite(params[k]) for k in ('low', 'high')) or params['high'] <= params['low'] or
                    params['clip'] != (key == 'input')):
                raise ValueError('invalid frozen scaling bounds')
            reference = (arrays[raw].astype(np.float32) - params['low']) / (params['high'] - params['low'])
            if params['clip']:
                reference = np.clip(reference, 0, 1)
            if arrays[scaled].dtype != np.float32 or not np.array_equal(reference, arrays[scaled]):
                raise ValueError('saved arrays do not preserve frozen scaling')
        if not np.array_equal(arrays['constant'], np.full_like(target, protocol['constant_value'])):
            raise ValueError('constant does not match the frozen training-derived value')
        arrays_by_source.append(arrays)
    for arm in ARMS:
        rows = report['fields'][arm]
        if [(r['source_id'], r['group_id']) for r in rows] != expected:
            raise ValueError('saved metric identities differ from locked fields')
        for row, artifact, arrays in zip(rows, report['artifacts'], arrays_by_source):
            if row['z'] != artifact['z'] or row['split'] != 'test':
                raise ValueError('saved metric split/depth mismatch')
            observed = _scores(arrays[arm], arrays['target'])
            if any(not _equal(row[k], value) for k, value in observed.items()):
                raise ValueError('saved metric does not match prediction pixels')
            if row['bright_pixel_count'] != int((arrays['target'] >= 1).sum()):
                raise ValueError('saved bright-pixel count mismatch')
        summary = report['summaries'][arm]
        groups = {g for _, g in expected}
        counts = {'field_count': len(expected), 'group_count': len(groups),
                  'undefined_pearson_fields': sum(r['pearson'] is None for r in rows),
                  'undefined_bright_fields': sum(r['bright_pixel_mae'] is None for r in rows)}
        if any(summary[k] != v for k, v in counts.items()):
            raise ValueError('saved group/undefined counts mismatch')
        for metric in ('mae', 'pearson', 'ssim', 'bright_pixel_mae'):
            values = [[r[metric] for r in rows if r['group_id'] == g and r[metric] is not None] for g in groups]
            value = float(np.mean([np.mean(v) for v in values if v])) if any(values) else None
            if not _equal(summary[metric], value):
                raise ValueError('saved metric aggregation mismatch')
    return report, arrays_by_source


def png_data(array, cmap):
    """Fixed [0,1] display only; saved float arrays/metrics remain unclipped."""
    rgba = colormaps[cmap](np.clip(array, 0, 1), bytes=True)
    stream = io.BytesIO()
    Image.fromarray(rgba).save(stream, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(stream.getvalue()).decode()


def export_viewer(report_path: Path, output_path: Path, *, expected_protocol_sha256: str):
    if output_path.exists():
        raise ValueError('viewer output already exists')
    report, arrays = verify_saved_evaluation(report_path, expected_protocol_sha256)
    payload = {'protocol_sha256': report['protocol_sha256'], 'report_sha256': _sha(report_path),
               'selected': report['validation_selected_context'], 'summaries': report['summaries'],
               'limitations': report['limitations'], 'models': report['protocol']['models'],
               'constant_value': report['protocol']['constant_value'], 'fields': []}
    for artifact, data in zip(report['artifacts'], arrays):
        source = artifact['source_id']
        payload['fields'].append({
            'source_id': source, 'group_id': artifact['group_id'], 'z': artifact['z'],
            'shape': list(data['target'].shape),
            'metrics': {arm: next(r for r in report['fields'][arm] if r['source_id'] == source) for arm in ARMS},
            'images': {'input': png_data(data['input'][1], 'gray'), 'target': png_data(data['target'], 'magma'),
                       'predictions': {a: png_data(data[a], 'magma') for a in ARMS},
                       'errors': {a: png_data(np.abs(data[a] - data['target']), 'inferno') for a in ARMS}},
            'outside_display_fraction': {a: float(((data[a] < 0) | (data[a] > 1)).mean())
                                         for a in ('target', *ARMS)}})
    # JSON-in-script escaping prevents report identifiers from closing script tags.
    encoded = json.dumps(payload, allow_nan=False).replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')
    template = files('virtual_staining').joinpath('viewer_template.html').read_text()
    document = template.replace('__EVIDENCE_JSON__', encoded)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('x') as stream:
        stream.write(document)
    return {'output_path': str(output_path), 'field_count': len(arrays), 'arm_count': 3,
            'source_report_sha256': payload['report_sha256'], 'output_sha256': _sha(output_path),
            'inference_performed': False, 'metrics_verified_from_saved_pixels': True,
            'display_range': [0, 1], 'protocol_sha256': report['protocol_sha256']}
