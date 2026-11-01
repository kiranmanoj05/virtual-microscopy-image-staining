# Reference baseline reset

## Purpose and current boundary

Question: can brightfield support useful nuclear localisation/counting on new
acquisition groups, and does predicted staining add value over direct analysis?
First reproduce credible image structure; count accuracy is a later endpoint
requiring independently checked reference masks. Neither colourful pictures
nor passing unit tests establish useful staining.

Phase one's negative evaluation remains frozen. This branch uses its ten
development fields only for pipeline validation. Its old test plate is excluded
from downloads and preparation. A new dataset with more independent plates and
a new frozen test set is required before a new final performance claim.

## Implemented changes

- Pinned, attributed Allen 3D architecture, including batch normalisation and
  learned down/up-sampling. It is not the historical compact 2D U-Net.
- All brightfield/DNA depths are loaded, not just a middle slice.
- OME physical voxel sizes drive cubic resampling to 0.29 micrometres isotropic.
- Each full input/target volume is z-scored independently **before** resampling,
  matching the transform order in the pinned `release_1/train_model.py`.
  Target statistics define the training/evaluation representation; inference
  never receives the target or its statistics. Output is relative structure,
  not calibrated raw fluorescence. No unknown target statistics may be used to
  reconstruct absolute fluorescence on a new sample.
- Original versioned receipts and hashes bind every raw development source.
  New arrays and metadata are independently fingerprinted. No old data/results
  are overwritten. Metadata/pixel provenance does not prove registration.

Upstream architecture source: `fd8fc92cc4c43d9d82da1c66c7e17e2447dd0830`,
`fnet/nn_modules/fnet_nn_3d_params.py`, SHA-256
`5e4a0bbbdf0d116956c28b08ad9c65cb3c6ea5a86cbb24b728d7eda586724f7c`.
Pinned paper release: `e1f51f50a641af048f76d71c9ce03a740ab33195`.
A local compatibility experiment loaded the paper-release state into the
vendored model strictly and obtained bitwise-identical output at widths 4 and
32 on a 16-cubed CPU input, seed 17. Width-32/depth-4 has 23,280,769 parameters.
This verifies architecture compatibility, not reproduction of published results.
See [third-party attribution](../THIRD_PARTY.md) for the upstream licence.

## Experiment settings declared before real training

First feasibility run: all eight original training fields and two validation
fields, 0.29-um full volumes, seed 17, width 8, depth 3, patch ZYX 16x64x64,
batch 2, Adam 0.001 with betas (0.5, 0.999), MSE, 1,000 updates. Validate every
250 updates using complete development volumes with tiled inference and a fixed
16-voxel halo. Select by group-balanced validation Pearson. Report MAE in the
new standardised space against zero; old MAE is not directly comparable.
Save every validation field and quarter/middle/three-quarter depth previews,
including failures. Display target/prediction on the same fixed [-1,3] range.

This reduced-width/depth/batch/step experiment is a compute feasibility pilot,
not the paper's training configuration. The reference architecture default is
width 32/depth 4; the paper baseline budget is 50,000 updates, batch 24,
patch 32x64x64. Measure compute before selecting a larger budget. Original OME
spacing replaces the paper CLI's hardcoded lateral scale; data release/splits
also differ. Do not describe the pilot as an exact paper reproduction.

Stop on non-finite loss/predictions or failed checksums. Do not choose slices,
reroll source splits, refit processing to improve held-out scores, or run old
test evaluation in this phase. Inspect learning curves and all failures before
selecting the next development experiment. No paid compute is configured.
