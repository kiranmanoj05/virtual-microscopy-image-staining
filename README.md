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

Review an Allen metadata CSV before choosing acquisition groups:

```bash
staining allen-inventory data/allen-metadata.csv --cell-line AICS-22 \
  --out-dir runs/allen-inventory
```

The inventory collapses cell rows into fields and uses plate IDs as provisional groups. It does not verify biological independence or download images. `staining split` freezes declared groups before patch extraction; run `staining split --help` for explicit held-out counts and seed options.

Prepare a small pilot and download its version-pinned files:

```bash
staining allen-pilot data/allen-metadata.csv --cell-line AICS-22 \
  --groups 6 --fields-per-group 2 --seed 17 --out-dir runs/allen-pilot
staining download-pilot runs/allen-pilot/pilot.json --out-dir data/raw/pilot
```

The plan records source versions, download sizes and whole-plate splits before image download. Check its reported size first. This is a pipeline pilot; plate independence and registration still require review. Downloads produce local checksum receipts and preserve incomplete files rather than silently replacing them.

`staining audit` checks file integrity and split overlap using the [manifest template](examples/manifest_template.csv). Raw data, checkpoints and run outputs are excluded from Git.

## References

Inspired by the Allen Institute's [label-free prediction work](https://github.com/AllenCellModeling/pytorch_fnet). The implementation is independent; reused components, if any, will be attributed explicitly.
