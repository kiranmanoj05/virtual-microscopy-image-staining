# First Allen microscopy sample

Updated 11 October 2026. This is an intake/inspection example, not a training or test set.

## Selection

The original label-free release offers structure archives. HTTP HEAD checks found `lamin_b1.tar.gz` at 22,208,784,318 bytes and `fibrillarin.tar.gz` at 24,458,579,242 bytes compressed. The first session instead uses one individually hosted field from the Allen Pipeline Integrated Cell dataset. This is a distinct data package; do not call it the original paper's exact training split.

The package README describes paired brightfield, DNA, cell membrane and tagged structure observations, and the official fnet example demonstrates retrieving individual fields from it. We inspected the live metadata and one versioned TIFF; the training implementation is independent.

Sources:
- Package README: https://allencell.s3.amazonaws.com/aics/pipeline_integrated_cell/README.md
- Metadata: https://allencell.s3.amazonaws.com/aics/pipeline_integrated_cell/metadata.csv
- Reference example: https://github.com/AllenCellModeling/pytorch_fnet/blob/master/examples/download_and_train.py
- Reuse terms: https://www.allencell.org/terms-of-use.html

The terms describe research/noncommercial use and attribution obligations. Keep raw images out of the code repository; record the applicable terms and citations before distributing examples or results.

## Inspected sample

- Logical path: `fovs/a1462124_3500001454_100X_20171023_1-Scene-01-P1-E04.ome.tiff`
- S3 version: `V7_cbkPaL_wJKMuHbgiCmxc2pCXdyOK6`
- Size: 377,743,516 bytes (about 378 MB).
- SHA-256: `5f6bcf749ee767cee0cc8a38e99dd626c89a455a7906dbff0fcfb074d97c9cf3`
- Metadata identities: FOVId 7649, WellId 67156, PlateId 3500001454, CellLine AICS-22, tagged structure Lysosome/LAMP1.
- TIFF series 0: axes `ZCYX`, shape `[70, 7, 624, 924]`, dtype `uint16`.

| Zero-based channel | OME label | Meaning recorded in OME |
| --- | --- | --- |
| 0 | CMDRP_1 | CellMask Deep Red plasma membrane stain |
| 1 | CMDRP | CellMask Deep Red plasma membrane stain |
| 2 | EGFP_1 | EGFP |
| 3 | EGFP | EGFP; associated dataset metadata identifies LAMP1 |
| 4 | H3342_1 | Hoechst 33342 |
| 5 | H3342 | Hoechst 33342; metadata's DNA channel |
| 6 | Bright_100X | Transmitted-light brightfield |

The channel numbers agree with this sample's CSV metadata (`ChannelNumberBrightfield=6`, `ChannelNumber405=5`, `ChannelNumberStruct=3`). Do not assume these indices hold for every file. The duplicated `_1` channels need investigation before any inclusion; metadata-only inspection has not established their role or pixel-level registration.

DNA is now the proposed first prediction target. This one field can exercise inspection/loading and tiny-subset debugging. It cannot support an independent generalisation estimate: its planes and crops all share the same field and well.

## Run on your Mac

From the existing repository with the virtual environment active:

```bash
git fetch origin
git switch main
git pull --ff-only
git switch --track origin/feat/data-intake
python -m pip install -e '.[ml]'
mkdir -p data/raw runs
```

Download only the versioned sample:

```bash
curl --fail --location --retry 3 \
  'https://allencell.s3.amazonaws.com/aics/pipeline_integrated_cell/fovs/a1462124_3500001454_100X_20171023_1-Scene-01-P1-E04.ome.tiff?versionId=V7_cbkPaL_wJKMuHbgiCmxc2pCXdyOK6' \
  --output data/raw/allen_first_fov.ome.tiff
shasum -a 256 data/raw/allen_first_fov.ome.tiff
staining inspect-tiff data/raw/allen_first_fov.ome.tiff > runs/allen-first-fov-info.json
python -m unittest discover -s tests -v
```

Compare the checksum above and share the small JSON inspection output. Expected: 14 passing checks when numpy and tifffile are installed. The inspector reads metadata without loading the whole pixel array. The complete inspection JSON stays in ignored `runs/`; the sample stays in ignored `data/`.

If you have already created a local `feat/data-intake` branch, switch to it and pull instead of creating it again. If `git status` shows changes, preserve them before switching branches.

## Next implementation gate

Read selected paired planes, render previews and inspect channel contents/alignment. Preserve bit depth; do not rescale training targets based on test target statistics. Select multiple fields and independent acquisition groups from verified metadata, then freeze splits before patch extraction. Use metadata identifiers explicitly rather than random image-slice splits. The single sample is not enough to choose split proportions or claim held-out performance.
