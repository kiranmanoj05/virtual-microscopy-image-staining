# Inspect paired brightfield and DNA planes

The sample checksum and OME metadata were reproduced on the user's Mac. Metadata intake PR #2 is merged; pixel loading and previews continue on `feat/paired-plane-loader`.

## Run locally

With your virtual environment active and the sample already downloaded:

```bash
git fetch origin
git switch --track origin/feat/paired-plane-loader
python -m pip install -e '.[ml]'
python -m pip freeze > runs/environment-mac.txt
python -m unittest discover -s tests -v
staining preview-pair data/raw/allen_first_fov.ome.tiff \
  --z 20 35 50 --input-channel 6 --target-channel 5 \
  --out-dir runs/paired-preview
open runs/paired-preview/preview.png
```

The dependency update adds `imagecodecs`: this TIFF uses LZW compression, and metadata inspection alone did not require decoding pixels. Expected result with the ML dependencies installed: 22 passing tests, a PNG and a JSON record in `runs/paired-preview/`.

## What to look for

Each row uses a different depth (zero-based z indices 20, 35 and 50). The first panel shows the measured brightfield input. The second shows the measured Hoechst DNA channel. The third displays the same target in green over a greyscale input. All panels come from real measurements; nothing is predicted by a model yet.

Compare recognisable structures across input and target, look for systematic shifts, blank/saturated channels and focus changes across depths. A single visual check cannot prove registration, identify all cellular structures or establish generalisation. Record any uncertainty rather than treating an attractive overlay as validation. The duplicated `_1` channels have not been adopted.

## Loader behaviour

`load_pair` reads the explicit input/target channels from the same TIFF series and z plane, decoding only two scalar YX pages. It maps indices from the actual named axes and supports C plus optional Z and singleton T before YX; it does not assume C always occupies the second dimension. It rejects unsupported layouts, multiple time points, invalid indices, identical input/target channels and non-finite intensities. Raw pixel values and dtype are preserved; no training normalisation, resizing or registration is performed.

For this file, the series is `[Z,C,Y,X] = [70,7,624,924]`. At z=35, input channel 6 maps to page 251 and target channel 5 to page 250. Each returned image is `[624,924]` with uint16 values. A three-depth preview decodes six pages, not all 490 pages in the stack.

## Display contract

The PNG uses 1st–99th percentile display bounds per channel, pooled over the selected depths. The same input bounds apply to every row, and the same target bounds apply to every row. This prevents every depth from independently appearing equally bright. The overlay uses a fixed greyscale/green blend.

Display bounds, source axes, page identities and selections are recorded in `preview.json`. These bounds are for visual inspection only and must not be reused as fitted training statistics. The arrays returned by the loader retain raw intensities. Constant display ranges render black.

## Verified and outstanding

All 22 local tests passed, including alternative channel-axis order, exact raw pixel comparisons, LZW decoding, invalid indices and consistent preview scaling. The command decoded/rendered the real sample at z=20,35,50 successfully, retaining uint16 values. Source planes contain finite nonconstant intensities. These software checks do not establish biological quality or registration. No model is trained.

The next data milestone needs representative fields from multiple independent acquisition groups, channel-content/registration assessment and a frozen split. Every slice and crop from this field must remain in one group. Its planes must not be randomly divided into train and test.

Source: Allen Institute for Cell Science, Pipeline Integrated Cell dataset. See [data intake provenance and terms](data-intake.md). Keep images and run outputs outside Git.
