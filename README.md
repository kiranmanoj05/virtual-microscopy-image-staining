# Virtual Microscopy Image Staining

An independent implementation of brightfield-to-fluorescence prediction, beginning with DNA fluorescence and a compact U-Net baseline.

## Status

Paired TIFF inspection, selected-plane loading and measured-image previews are implemented. No model has been trained and no prediction performance has been measured.

## Setup

Requires Python 3.11–3.13.

```bash
python -m pip install -e '.[ml]'
python -m unittest discover -s tests -v
staining doctor
```

Inspect a local microscopy file and preview selected paired planes:

```bash
staining inspect-tiff data/raw/sample.ome.tiff
staining preview-pair data/raw/sample.ome.tiff \
  --z 20 35 50 --input-channel 6 --target-channel 5 \
  --out-dir runs/paired-preview
```

Channel indices are sample-specific; verify their identities before use. Preview panels show measured data, not predictions. Alignment must be checked before training.

`staining audit` checks file integrity and split overlap using the [manifest template](examples/manifest_template.csv). Raw data, checkpoints and run outputs are excluded from Git.

## References

Inspired by the Allen Institute's [label-free prediction work](https://github.com/AllenCellModeling/pytorch_fnet). The implementation is independent; reused components, if any, will be attributed explicitly.
