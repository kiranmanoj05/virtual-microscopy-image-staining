# Virtual Microscopy Image Staining

An independent implementation of brightfield-to-fluorescence prediction, beginning with DNA fluorescence and a compact U-Net baseline.

## Status

Paired TIFF inspection, selected-plane loading and measured-image previews are implemented. A compact U-Net and one-patch debugging command are implemented. One-patch microscopy learning has been checked locally. A pilot trainer now supports full-plane validation and checkpoint recovery; held-out test performance has not been measured.

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

Audit downloaded pilot files and export a training-only preview:

```bash
staining audit-pilot runs/allen-pilot/pilot.json --data-dir data/raw/pilot \
  --out-dir runs/pilot-audit
```

The audit verifies receipts, bytes, OME channel identities and supported layouts. It decodes a fixed set of depths in training/validation fields and leaves test pixels unopened during development. Inspect the training preview before adopting a training depth policy; a successful software audit does not establish registration.

Prepare the fixed middle-plane baseline and check learning on one training patch:

```bash
staining prepare-training runs/pilot-audit/audit.json --out-dir data/processed/middle-plane
staining overfit data/processed/middle-plane/dataset.json --steps 100 \
  --out-dir runs/overfit-check
```

Scaling is fitted on training fields only. Input values are clipped to the training percentile range; targets use an affine transform without clipping. The U-Net has one continuous output channel. The overfit command is a debugging check on one fixed training patch, not a held-out performance result. Test pixels remain unopened.

Inspect the saved debug model against its measured training patch:

```bash
staining preview-overfit data/processed/middle-plane/dataset.json \
  runs/overfit-check/debug-model.pt --out-dir runs/overfit-preview
```

The preview uses matching signal display ranges and reports unclipped patch MAE/Pearson alongside the same training patch's median baseline. Constant-image Pearson is undefined. These are debugging comparisons on training pixels, not held-out evaluation.

Train across the prepared training fields with full-plane validation:

```bash
staining train data/processed/middle-plane/dataset.json --steps 500 \
  --batch-size 2 --patch-size 128 --validate-every 100 \
  --out-dir runs/pilot-training
```

Whole-plane validation reports group-balanced MAE, Pearson and SSIM against a training-derived median and an untrained model. SSIM uses a fixed training-scaled range of 1 and window 7; metrics use unclipped target values. The primary checkpoint criterion is validation MAE. Test fields remain excluded. Both best and last checkpoints are saved with optimiser/RNG state; `--resume runs/pilot-training/last-model.pt` continues a completed run in a new output directory with `--steps` interpreted as the total desired steps.

Diagnose an existing training run before changing its experiment settings:

```bash
staining diagnose-training data/processed/middle-plane/dataset.json \
  runs/pilot-training --out-dir runs/pilot-diagnostics
```

This compares both saved checkpoints on training and validation planes, reports intensity/prediction distributions and checks fixed crop predictions against the same pixels in full-image predictions. A diagnostic bright-target threshold is fixed at the training-target 99th percentile. It changes no experiment settings and reads no test images.

Inspect development data without a model checkpoint:

```bash
staining diagnose-data data/processed/middle-plane/dataset.json \
  --patch-size 128 --out-dir runs/data-diagnostics
```

The report preserves raw and pre-clipping intensity distributions, measures values outside the training-fitted range, and compares signal in fixed centre/corner crops. Its constant and bright threshold come from training data only. Summaries weight acquisition groups equally. Fixed crops are descriptive probes, not estimates of random training crop prevalence; intensity shifts alone do not establish their cause. Test entries and changed pixels are rejected.

`staining audit` checks file integrity and split overlap using the [manifest template](examples/manifest_template.csv). Raw data, checkpoints and run outputs are excluded from Git.

Compare a duration-only continuation after both runs finish:

```bash
staining compare-duration runs/pilot-training runs/pilot-training-2000 \
  --out-dir runs/duration-comparison
```

This checks checkpoint hashes, continuation identity, preserved history, source/depth identities, group-balanced summaries and equal recorded settings, environment and source revision. It reports paired per-field and summary metric differences without opening pixels. Validation cadence cannot change during resume. Best checkpoints are selected over different training budgets; MAE improvement alone does not establish useful staining. This strict command is not for cross-device or multi-factor comparisons.

For an objective-only experiment, run two fresh trainers with identical settings and
`--objective l1` (the default) versus `--objective mse`, then compare:

```bash
staining compare-objectives runs/fresh-l1 runs/fresh-mse --out-dir runs/objective-comparison
```

Both runs must use the same source, budget, data, environment and seeded initial weights.
The comparison also checks fingerprints of the actual sampled training batches. Warm starts
and additional configuration differences are rejected. Resume cannot change objectives;
legacy checkpoints without an objective field are treated as L1 only. Selection and evaluation
remain validation MAE/Pearson/SSIM against the same training-derived median, regardless of
training objective. L1 and MSE training-loss values are not directly comparable.

## References

Inspired by the Allen Institute's [label-free prediction work](https://github.com/AllenCellModeling/pytorch_fnet). The implementation is independent; reused components, if any, will be attributed explicitly.
