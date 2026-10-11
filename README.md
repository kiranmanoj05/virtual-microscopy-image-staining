# Virtual Microscopy Image Staining

Predicting fluorescence images from paired brightfield microscopy, with a focus on reproducible evaluation and inspecting where predictions fail.

## Status

Started 11 October 2026. The foundation provides setup, environment diagnostics and paired-file integrity/split checks. Data intake now includes TIFF metadata inspection and a verified single-field sample route. No model has been trained and no microscopy performance has been measured.

## Plan

Start with one target and a compact 2D U-Net. Verify pairing and acquisition-group splits before training. Compare predictions with simple baselines on held-out images, then investigate whether neighbouring-plane context improves results. A visual explorer will show input, measured target, prediction and error together.

## Getting started

See [the first local session](docs/getting-started.md) for installation and commands. The lightweight foundation needs Python 3.11–3.13:

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
staining doctor
```

Use `python -m pip install -e '.[ml]'` when preparing the microscopy/ML environment.

## Documentation

- [Technical specification and milestone estimates](docs/specification.md)
- [Local setup and manifest auditing](docs/getting-started.md)
- [Manifest column template](examples/manifest_template.csv)
- [First microscopy sample and channel inspection](docs/data-intake.md)
- [Paired-plane loading and visual inspection](docs/paired-preview.md)

## References

The Allen Institute's [label-free prediction work](https://github.com/AllenCellModeling/pytorch_fnet) is a methodological reference. Its [image release](https://downloads.allencell.org/publication-data/label-free-prediction/index.html) is a dataset candidate; adoption depends on verified terms, pairing and metadata. This repository implements its own pipeline and records any future reused components explicitly.

Raw images, checkpoints and large experiment outputs are excluded from Git. This project is separate from ORB.
