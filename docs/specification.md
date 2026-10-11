# Project specification

Updated: 11 October 2026. Status: foundation implemented; dataset not selected.

## Question

Can a small model predict a measured fluorescence channel from paired brightfield images on previously unseen acquisition groups? Does neighbouring-plane context improve that prediction enough to justify its extra compute?

This is an independently implemented learning project. It draws on published virtual staining work, with attribution, rather than porting an existing repository. The contribution is a controlled comparison and clear inspection of errors; scientific novelty is not assumed.

## First deliverable

A reproducible, single-target experiment: audited pairs, a frozen group split, a compact 2D U-Net, saved training configuration, held-out predictions, Pearson/SSIM/MAE results, baseline comparisons and representative failures. A small visual explorer follows this working pipeline.

## Scope

- Begin with one fluorescence target whose identity and alignment can be verified.
- Use a compact 2D PyTorch model and patches to keep memory demand manageable.
- Keep all planes, channels and patches from a source field in one split. Group related fields at the highest dependency level supported by metadata.
- Compare a single brightfield plane with a small stack of neighbouring brightfield planes if the dataset supports it. Predict the same target plane and use identical split identities and evaluation settings.
- Add input, measured target, predicted target and absolute-error panels with consistent display ranges.
- Consider multi-target predictions only after the single-target result is reproducible. Combining channels from separate cell lines cannot be treated as a measured multi-channel image of the same cell.

Full 3D models, GANs, cloud infrastructure and clinical validation are outside the first version. They may be reconsidered after the baseline, with a concrete benefit and cost.

## Dataset decision gate

The Allen Institute label-free prediction release is a candidate, not an adopted dataset. The official download page lists structure-specific image archives and links to reuse terms. Before downloading a substantial archive, inspect its size, terms, formats and available metadata. Start with a small selection of independent fields for one structure.

Record release URL, access date, licence/reuse terms, citations, file checksums, channel identities, dimensions/axes, voxel spacing if available, sample/well/acquisition identifiers and pairing evidence. Verify actual intensity images and registration visually; filenames and matching shapes alone are insufficient.

If metadata cannot support specimen-level independence, state the narrower evaluation claim. Do not invent acquisition identifiers.

## Data and split contract

The source inventory contains one row per field with verified `source_id` and `group_id` values. The split command requires explicit validation/test group counts, assigns complete groups using a stable seeded hash order, records a fingerprint and refuses to overwrite an existing plan. Its deterministic output does not prove that the supplied group identities are biologically independent.

The paired-image CSV audit requires `pair_id`, `input_path`, `target_path`, `source_id`, `group_id`, `structure`, `split`, `input_sha256` and `target_sha256`. Paths are relative to a local data root. Valid split names are `train`, `validation` and `test`.

The audit rejects missing required values, duplicate pair IDs, missing files, checksum changes, paths outside the data root, source/group overlap between splits and identical file bytes across splits. Its output is a bookkeeping check; it does not decode images, check image axes or establish registration. Pixel-equivalent images saved with different metadata/compression may evade file-byte duplicate checks. Decoded-image checks belong in the next milestone.

Freeze assignments at the group/source level before extracting patches. Choose split proportions only after counting independent groups; a 70/15/15 ratio is a candidate, not a requirement. Prevent leakage through related wells, overlapping crops, neighbouring planes, augmentation and fitted preprocessing. Reserve the test split until experiment choices are fixed on validation data.

## Proposed baseline

Begin with a small U-Net, one input channel and one continuous output channel. Candidate settings: 256-pixel patches, starting width 16, L1 regression loss and Adam. These settings are proposals until a data audit and memory/runtime benchmark support them. Fit any dataset-level scaling on training data only; reuse it during validation and inference. Apply the same spatial augmentations to each paired input and target.

Before a full run, overfit a tiny training subset to check gradients, target handling and output shapes. Use a short local pilot to estimate seconds per step, memory and projected training time. Record seeds, exact package versions, device, Git commit, split fingerprint, configuration and checkpoint hash. Resolve and record an environment lock before the first reported training experiment; the current optional dependency ranges are installation bounds, not a locked experiment environment.

## Evaluation

- Compare with an untrained model and a training-derived constant-intensity baseline. Add a training-mean spatial image only if image geometry makes that comparison meaningful.
- Calculate Pearson, SSIM and MAE per held-out source image and target channel, after explicitly specified patch reconstruction where needed. Label any patch-only metrics clearly.
- Pearson is undefined for constant arrays. Count and report undefined observations rather than substituting a favourable score.
- Use a fixed, documented SSIM intensity range and window size. Do not choose a fresh range from each test image.
- Report valid image counts, excluded cases, distributions and group-balanced summaries. Use group-level uncertainty estimates only when the number of independent groups supports them.
- Choose representative and failed predictions using a documented selection method, with matching display scales. Include error maps rather than relying on colourful composites.

A trained model must beat the chosen baseline on the primary validation metric before progressing to optional extensions. A weak or negative test result is still a valid outcome and must be reported.

## Milestones and estimated effort

These are planning estimates, not promised completion dates. Training and downloads are additional elapsed time.

| Milestone | Deliverable | Active work |
| --- | --- | --- |
| Foundation | setup, environment doctor, manifest checks, tests and CI | 1–2 hours |
| Data intake | candidate decision, metadata audit, paired previews, frozen split | 3–5 hours |
| Single-target baseline | image loader, U-Net, training/resume, tiny-subset check | 5–8 hours |
| Evaluation | deterministic baselines, metrics, full-image inference, failure gallery | 4–7 hours |
| Context experiment | single-plane versus neighbouring-plane comparison | 4–7 hours |
| Delivery | visual explorer, reproducible commands and evidence-backed report | 3–6 hours |

Total: roughly 20–35 hours, or 4–6 weeks at about 45–60 minutes on most days. Dataset access and local compute could change this. Multi-target work is optional and outside this estimate. No paid compute is required or activated by this plan.

## Version control

`main` holds coherent, checked milestones. Each feature uses a short-lived branch and a PR explaining its purpose, behaviour and verification. Commit at real boundaries; do not manufacture activity, backdate work or add deliberate mistakes to simulate development.

Suggested later branches: `feat/data-intake`, `feat/unet-baseline`, `feat/evaluation`, `experiment/plane-context`, `feat/prediction-viewer`. Example messages: `validate paired image metadata`, `train a compact unet on one target`, `report held-out errors alongside baseline scores`.

Private project reports and the personal learning guide remain outside Git. Data, credentials, checkpoints and large run outputs stay outside source control. Keep small configs and split identities in Git where their terms permit it.

## Current implementation evidence

The foundation implements packaging, `staining doctor`, `staining audit`, a manifest template, 11 automated integrity/leakage checks and a GitHub Actions workflow. Local verification used Python 3.12 on Linux. No microscopy files were downloaded, image alignment checked, model trained or performance score measured. MPS is not verified until the user runs the environment check on their Mac.

## References

- Ounkomol et al. (2018), label-free prediction: https://doi.org/10.1038/s41592-018-0111-2
- Allen Institute reference implementation (archived): https://github.com/AllenCellModeling/pytorch_fnet
- Official dataset page: https://downloads.allencell.org/publication-data/label-free-prediction/index.html
- Dataset terms: https://www.allencell.org/terms-of-use.html
- Alternative reference: https://github.com/WayScience/virtual_stain_flow
- PyTorch MPS guidance: https://docs.pytorch.org/docs/stable/notes/mps.html

These are references and candidates; the user's original inspiration is still unconfirmed.
